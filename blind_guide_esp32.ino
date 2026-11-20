/*
  BLIND GUIDE  -  ESP32 side

  What this program does
    1. Listens to the MaixCam on a UART. Two kinds of lines arrive (see scene.h):
         $CUE,left|straight|right|none*HH   about 10 times per second  -> the 3 vibration motors
         $SCN;L,P,nuel,-1;F,O,table,25*HH   what / who is in front    -> kept for the button
    2. Vibration motors (LEFT / CENTRE / RIGHT): the motor that does NOT vibrate is the free way.
         cue straight -> LEFT + RIGHT vibrate
         cue left     -> CENTRE + RIGHT vibrate
         cue right    -> LEFT + CENTRE vibrate
       If no cue arrives for 0.5 s, all three motors blink (= "camera link lost").
    3. Button: press once = start talking, press again = stop. While talking it keeps saying the
       LATEST scene, e.g. "left nuel, front carmen, right table two point five meters".
    4. Audio: the ESP32 has no audio pin, but it IS a Bluetooth audio transmitter (A2DP source).
       The words are stored inside the program (clips.h), the ESP32 glues them into a sentence
       and streams the sound to the Bluetooth earphone.

  Arduino IDE settings:  Board "ESP32 Dev Module"   Tools > Partition Scheme > "Huge APP (3MB No OTA/1MB SPIFFS)"
  Library needed:        ESP32-A2DP  (by Phil Schatzmann)
  Wiring:                MaixCam U2T (TX pin on its header) -> D33 (GPIO33),  GND <-> GND,
                         motors LEFT/CENTRE/RIGHT -> GPIO19/26/27 (through a transistor each),
                         button between GPIO14 and GND.
*/
#include <Arduino.h>
#include <vector>
#include "BluetoothA2DPSource.h"
#include "scene.h"
#include "clips.h"

// ============================== SETTINGS ==============================
// The earphone's Bluetooth name. Put it in pairing mode and look at the name your phone shows.
// Several spellings are tried; the first one that is found is used.
static std::vector<const char*> EARPHONE_NAMES = {
  "kiip DTS 10", "KIIP DTS10", "kiip DTS10", "DTS10", "DTS 10", "KIIP DTS 10"
};
#define VOLUME           127      // 0..127 (127 = full). Loudness is set by the earphone's own volume.

#define MAIX_RX_PIN      33       // ESP32 RX  <- MaixCam A21 (TX).  33 = the pin labelled D33
#define MAIX_TX_PIN      17       // ESP32 TX  -> MaixCam A22 (RX), not used yet
#define MAIX_BAUD        115200   // same as BAUD in main.py

#define PIN_MOTOR_LEFT   19
#define PIN_MOTOR_CENTRE 26
#define PIN_MOTOR_RIGHT  27
#define PIN_BUTTON       14       // button to GND (internal pull-up is used)

#define DEBUG_MAIX       1        // 1 = every 3 s print how much data came from the MaixCam (0 = quiet)
#define CUE_TIMEOUT_MS   500      // no cue for this long -> motors blink "link lost"
#define SCENE_TIMEOUT_MS 2000     // a scene older than this is too old to read out
#define SPEECH_GAP_MS    700      // pause between two announcements
#define LEAD_MS          150      // silence before a sentence (earphones swallow the first sound)
#define PAUSE_MS         350      // silence between two things in a sentence

// Which motors vibrate for each cue.            LEFT CENTRE RIGHT
static const uint8_t PATTERN[4][3] = {
  {0, 0, 0},   // none
  {0, 1, 1},   // left      -> the free way is LEFT, so LEFT stays quiet
  {1, 0, 1},   // straight  -> the free way is CENTRE
  {1, 1, 0},   // right
};
// =======================================================================

static const uint8_t MOTOR_PINS[3] = {PIN_MOTOR_LEFT, PIN_MOTOR_CENTRE, PIN_MOTOR_RIGHT};

BluetoothA2DPSource a2dp;

// ----------------------------------------------------------------------
//  AUDIO PLAYER
//  The A2DP library calls audioCb() all the time and asks for sound: 44100 Hz, stereo, 16 bit.
//  Our clips are 22050 Hz mono, so every clip sample is used twice (the one in between is the
//  average of its neighbours) and copied to left + right. When nothing is playing we hand over
//  silence (zeros), which keeps the Bluetooth link "streaming" so the first word is not lost.
// ----------------------------------------------------------------------
struct ClipRef { const int16_t* data; uint32_t len; };   // data == nullptr -> silence of len samples

#define MAX_LIST 64
struct Player {
  ClipRef  list[MAX_LIST];
  int      listLen = 0;
  int      listIdx = 0;
  uint32_t pos     = 0;       // sample inside the current clip (22050 Hz)
  bool     phase   = false;   // false: output the sample itself, true: output the middle value
  volatile bool playing = false;   // read by loop(), changed by audioCb()
} player;

static portMUX_TYPE playerMux = portMUX_INITIALIZER_UNLOCKED;   // audioCb and loop() run on different cores

static void playerNextItem() {              // called with the lock held
  player.listIdx++;
  player.pos = 0;
  player.phase = false;
  if (player.listIdx >= player.listLen) player.playing = false;
}

static inline int16_t playerSample() {      // one output sample at 44100 Hz
  if (!player.playing) return 0;
  const ClipRef& c = player.list[player.listIdx];
  int16_t a = 0, b = 0;
  if (c.data) {
    a = c.data[player.pos];
    b = (player.pos + 1 < c.len) ? c.data[player.pos + 1] : 0;
  }
  int16_t s = player.phase ? (int16_t)(((int32_t)a + (int32_t)b) / 2) : a;
  if (player.phase) {
    player.phase = false;
    if (++player.pos >= c.len) playerNextItem();
  } else {
    player.phase = true;
  }
  return s;
}

// The function the library calls.  data = interleaved left/right 16-bit samples.
static int32_t audioCb(uint8_t* data, int32_t byteCount) {
  int16_t* out = (int16_t*)data;
  int32_t frames = byteCount / 4;           // 1 frame = left + right = 4 bytes
  portENTER_CRITICAL(&playerMux);
  for (int32_t i = 0; i < frames; i++) {
    int16_t s = playerSample();
    out[2 * i]     = s;
    out[2 * i + 1] = s;
  }
  portEXIT_CRITICAL(&playerMux);
  return frames * 4;
}

static const ClipEntry* findClip(const char* name) {
  for (int i = 0; i < CLIP_COUNT; i++)
    if (strcmp(CLIPS[i].name, name) == 0) return &CLIPS[i];
  return nullptr;
}

static bool hasClip(const char* name) { return name[0] == '_' || findClip(name) != nullptr; }

static void stopSpeech() {
  portENTER_CRITICAL(&playerMux);
  player.playing = false;
  portEXIT_CRITICAL(&playerMux);
}

// words = clip names ("_lead" and "_gap" are silences). Starts playing them one after another.
static void startSpeech(char words[][NAME_LEN], int n) {
  ClipRef tmp[MAX_LIST];
  int m = 0;
  for (int i = 0; i < n && m < MAX_LIST; i++) {
    if (!strcmp(words[i], "_lead"))      tmp[m++] = {nullptr, (uint32_t)CLIP_RATE * LEAD_MS / 1000};
    else if (!strcmp(words[i], "_gap"))  tmp[m++] = {nullptr, (uint32_t)CLIP_RATE * PAUSE_MS / 1000};
    else {
      const ClipEntry* c = findClip(words[i]);
      if (c) tmp[m++] = {c->data, c->len};
      else Serial.printf("missing clip: %s\n", words[i]);
    }
  }
  if (m == 0) return;
  portENTER_CRITICAL(&playerMux);
  memcpy(player.list, tmp, sizeof(ClipRef) * m);
  player.listLen = m;
  player.listIdx = 0;
  player.pos = 0;
  player.phase = false;
  player.playing = true;
  portEXIT_CRITICAL(&playerMux);
}

// ----------------------------------------------------------------------
//  MAIXCAM LINES -> latest cue / latest scene
// ----------------------------------------------------------------------
static Cue      cue = CUE_NONE;
static bool     haveCue = false;
static uint32_t lastCueMs = 0;
static Scene    scene;
static bool     haveScene = false;
static uint32_t sceneMs = 0;
static uint32_t badLines = 0;
static uint32_t goodLines = 0;
static uint32_t rxBytes = 0;     // everything that arrived from the MaixCam (for the debug line)

static void handleLine(const char* line) {
  char body[200];
  if (!checkLine(line, body, sizeof(body))) { badLines++; return; }   // damaged line: ignore it
  goodLines++;
  Cue c;
  Scene s;
  if (parseCue(body, c)) {
    if (!haveCue || c != cue) Serial.printf("cue -> %d\n", (int)c);
    cue = c; haveCue = true; lastCueMs = millis();
  } else if (parseScene(body, s)) {
    scene = s; haveScene = true; sceneMs = millis();
  } else {
    badLines++;
  }
}

static void readMaix() {
  static char buf[256];
  static size_t len = 0;
  static bool overflow = false;
  while (Serial2.available()) {
    char ch = (char)Serial2.read();
    rxBytes++;
    if (ch == '\n') {
      if (!overflow) { buf[len] = 0; handleLine(buf); }
      len = 0; overflow = false;
    } else if (ch != '\r') {
      if (len < sizeof(buf) - 1) buf[len++] = ch; else overflow = true;
    }
  }
}

// ----------------------------------------------------------------------
//  MOTORS
// ----------------------------------------------------------------------
static void updateMotors() {
  uint32_t now = millis();
  if (!haveCue || now - lastCueMs > CUE_TIMEOUT_MS) {
    bool on = (now % 1000) < 120;                       // all blink: no camera link
    for (int i = 0; i < 3; i++) digitalWrite(MOTOR_PINS[i], on ? HIGH : LOW);
    return;
  }
  for (int i = 0; i < 3; i++) digitalWrite(MOTOR_PINS[i], PATTERN[cue][i] ? HIGH : LOW);
}

// ----------------------------------------------------------------------
//  BUTTON + SPEECH
// ----------------------------------------------------------------------
static bool     speaking = false;       // true = keep announcing
static uint32_t lastSpeechEnd = 0;

static void onButtonPress() {
  speaking = !speaking;
  stopSpeech();
  lastSpeechEnd = 0;                    // start at once when switched on
  Serial.println(speaking ? "button: talking ON" : "button: talking OFF");
}

static void handleButton() {
  static bool lastRead = HIGH, stable = HIGH;
  static uint32_t changedAt = 0;
  uint32_t now = millis();
  bool r = digitalRead(PIN_BUTTON);
  if (r != lastRead) { lastRead = r; changedAt = now; }
  if (now - changedAt > 30 && r != stable) {            // 30 ms debounce
    stable = r;
    if (stable == LOW) onButtonPress();
  }
}

static void sayScene(const Scene& sc) {
  static char words[64][NAME_LEN];
  int n = buildPhrase(sc, hasClip, words, 64);
  Serial.print("say:");
  for (int i = 0; i < n; i++) { Serial.print(' '); Serial.print(words[i]); }
  Serial.println();
  startSpeech(words, n);
}

// TEST WITHOUT THE CAMERA: type  t  in the Serial Monitor message box and press Enter.
// The ESP32 then says: "left nuel, front carmen, right table two point five meters".
static void handleSerialTest() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch != 't' && ch != 'T') continue;
    if (!a2dp.is_connected()) { Serial.println("test: earphone is not connected yet"); continue; }
    Scene t;
    t.count = 3;
    t.items[0] = {'L', true,  "nuel",   -1};
    t.items[1] = {'F', true,  "carmen", -1};
    t.items[2] = {'R', false, "table",  25};
    Serial.println("test sentence:");
    sayScene(t);
  }
}

static void updateSpeech() {
  static bool wasPlaying = false;
  uint32_t now = millis();
  bool playing = player.playing;
  if (wasPlaying && !playing) lastSpeechEnd = now;
  wasPlaying = playing;

  if (!speaking || playing) return;
  if (now - lastSpeechEnd < SPEECH_GAP_MS) return;
  if (!a2dp.is_connected()) return;                      // nobody to talk to
  if (!haveScene || now - sceneMs > SCENE_TIMEOUT_MS) return;   // no fresh camera information

  sayScene(scene);
  if (!player.playing) lastSpeechEnd = now;              // nothing could be played: wait before retrying
}

// ----------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 3; i++) { pinMode(MOTOR_PINS[i], OUTPUT); digitalWrite(MOTOR_PINS[i], LOW); }
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  Serial2.begin(MAIX_BAUD, SERIAL_8N1, MAIX_RX_PIN, MAIX_TX_PIN);

  a2dp.set_auto_reconnect(true);          // if the earphone goes away, connect again by itself
  a2dp.set_data_callback(audioCb);        // the library asks us for sound
  a2dp.set_volume(VOLUME);
  a2dp.start(EARPHONE_NAMES);             // look for the earphone and connect

  Serial.printf("ready, %d words stored. Looking for the earphone...\n", CLIP_COUNT);
}

void loop() {
  readMaix();                             // 1) lines from the MaixCam
  updateMotors();                         // 2) cue -> motors (every pass = lowest delay)
  handleButton();                         // 3) button
  updateSpeech();                         // 4) say the latest scene when talking is ON
  handleSerialTest();                     // 4b) type t in the Serial Monitor = say a test sentence

  static bool wasConnected = false;       // 5) tell the serial monitor what Bluetooth is doing
  bool c = a2dp.is_connected();
  if (c != wasConnected) {
    wasConnected = c;
    Serial.printf("[%lus] %s\n", (unsigned long)(millis() / 1000), c ? "earphone CONNECTED" : "earphone disconnected");
  }

#if DEBUG_MAIX
  static uint32_t lastDbg = 0;
  if (millis() - lastDbg >= 3000) {
    lastDbg = millis();
    if (rxBytes == 0)
      Serial.printf("[%lus] MaixCam: NO DATA (wire / pin / app not running)\n", (unsigned long)(millis() / 1000));
    else
      Serial.printf("[%lus] MaixCam: bytes=%lu good lines=%lu damaged lines=%lu\n",
                    (unsigned long)(millis() / 1000), (unsigned long)rxBytes,
                    (unsigned long)goodLines, (unsigned long)badLines);
  }
#endif
  delay(1);
}
