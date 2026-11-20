// scene.h  -  the "brain" that has NO Arduino code in it, so it can be tested on a PC.
//
// 1. checkLine()   : is a line from the MaixCam intact?  ($BODY*HH, HH = XOR of BODY)
// 2. parseCue()    : "CUE,left"  ->  CUE_LEFT
// 3. parseScene()  : "SCN;L,P,nuel,-1;F,O,table,25"  ->  a list of things in front
// 4. buildPhrase() : list of things  ->  list of audio clip names to play
//
// Example:  scene = nuel on the left, carmen in front, a table 2.5 m on the right
//           phrase = left nuel / front carmen / right table two point five meters
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SPEAK_PERSON_DISTANCE
#define SPEAK_PERSON_DISTANCE 0   // 1 = also say the distance of people ("front carmen two meters")
#endif

#define NAME_LEN 20               // longest clip name (19 letters + end marker)
#define MAX_ITEMS 6               // most things kept from one scene line

enum Cue : uint8_t { CUE_NONE = 0, CUE_LEFT = 1, CUE_STRAIGHT = 2, CUE_RIGHT = 3 };

struct Item {
  char side;                      // 'L' left, 'F' front, 'R' right
  bool person;                    // true = a person (name or "person"), false = an object
  char label[NAME_LEN];           // clip name: "nuel", "carmen", "person", "table" ...
  int  dm;                        // distance in tenths of a metre, -1 = unknown
};

struct Scene {
  Item    items[MAX_ITEMS];
  uint8_t count;
};

// ------------------------------------------------------------------ 1. checksum
static inline uint8_t xorSum(const char* s, size_t n) {
  uint8_t c = 0;
  for (size_t i = 0; i < n; i++) c ^= (uint8_t)s[i];
  return c;
}

static inline int hexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

// line = "$BODY*HH" (without the newline). Returns true only if the checksum matches.
// On success BODY is copied into body[].
static bool checkLine(const char* line, char* body, size_t bodyMax) {
  size_t len = strlen(line);
  if (len < 5 || line[0] != '$') return false;
  const char* star = line + len - 3;                  // "*HH" must be the last 3 characters
  if (star[0] != '*') return false;
  int hi = hexVal(star[1]), lo = hexVal(star[2]);
  if (hi < 0 || lo < 0) return false;
  size_t bodyLen = (size_t)(star - (line + 1));
  if (bodyLen + 1 > bodyMax) return false;
  if (xorSum(line + 1, bodyLen) != (uint8_t)(hi * 16 + lo)) return false;
  memcpy(body, line + 1, bodyLen);
  body[bodyLen] = 0;
  return true;
}

// ------------------------------------------------------------------ 2. the cue
static bool parseCue(const char* body, Cue& cue) {
  if (strncmp(body, "CUE,", 4) != 0) return false;
  const char* v = body + 4;
  if      (!strcmp(v, "left"))     cue = CUE_LEFT;
  else if (!strcmp(v, "straight")) cue = CUE_STRAIGHT;
  else if (!strcmp(v, "right"))    cue = CUE_RIGHT;
  else if (!strcmp(v, "none"))     cue = CUE_NONE;
  else return false;
  return true;
}

// ------------------------------------------------------------------ 3. the scene
// body = "SCN" followed by items, each item = ";side,kind,label,dm"
//   side  L / F / R        kind  P (person) / O (object)
//   label a-z 0-9 _        dm    distance in tenths of a metre, or -1
// A scene is only accepted when EVERY item is well formed.
static bool parseScene(const char* body, Scene& out) {
  if (strncmp(body, "SCN", 3) != 0) return false;
  Scene tmp;
  tmp.count = 0;
  const char* p = body + 3;
  while (*p == ';') {
    p++;
    const char* end = strchr(p, ';');
    size_t len = end ? (size_t)(end - p) : strlen(p);
    char item[48];
    if (len == 0 || len >= sizeof(item)) return false;
    memcpy(item, p, len);
    item[len] = 0;

    char* f[4];
    int nf = 0;
    f[nf++] = item;
    for (char* c = item; *c; ++c) {
      if (*c == ',') {
        *c = 0;
        if (nf >= 4) return false;
        f[nf++] = c + 1;
      }
    }
    if (nf != 4) return false;

    char side = f[0][0];
    if ((side != 'L' && side != 'F' && side != 'R') || f[0][1] != 0) return false;
    char kind = f[1][0];
    if ((kind != 'P' && kind != 'O') || f[1][1] != 0) return false;
    size_t ll = strlen(f[2]);
    if (ll == 0 || ll >= NAME_LEN) return false;
    for (size_t i = 0; i < ll; i++) {
      char ch = f[2][i];
      if (!((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_')) return false;
    }
    char* endp = nullptr;
    long dm = strtol(f[3], &endp, 10);
    if (endp == f[3] || *endp != 0 || dm < -1 || dm > 9999) return false;

    if (tmp.count < MAX_ITEMS) {
      Item& it = tmp.items[tmp.count++];
      it.side = side;
      it.person = (kind == 'P');
      strcpy(it.label, f[2]);
      it.dm = (int)dm;
    }
    p += len;
  }
  if (*p != 0) return false;
  out = tmp;
  return true;
}

// ------------------------------------------------------------------ 4. the sentence
typedef bool (*HasClipFn)(const char* name);

static void addWord(char out[][NAME_LEN], int& n, int maxOut, const char* w) {
  if (n >= maxOut) return;
  strncpy(out[n], w, NAME_LEN - 1);
  out[n][NAME_LEN - 1] = 0;
  n++;
}

static const char* const DIGIT_WORD[10] = {"zero", "one", "two", "three", "four",
                                           "five", "six", "seven", "eight", "nine"};

// distance in tenths of a metre -> words, rounded to the nearest half metre:
//   25 -> two point five meters      30 -> three meters      10 -> one meter
static void addDistance(char out[][NAME_LEN], int& n, int maxOut, int dm) {
  int h = (dm + 2) / 5;                  // half-metres, rounded
  if (h < 1) h = 1;                      // never say "zero meters"
  if (h > 19) return;                    // 10 m or more: don't say a number
  addWord(out, n, maxOut, DIGIT_WORD[h / 2]);
  if (h % 2) {
    addWord(out, n, maxOut, "point");
    addWord(out, n, maxOut, "five");
  }
  addWord(out, n, maxOut, h == 2 ? "meter" : "meters");
}

static int sideRank(char s) { return s == 'L' ? 0 : (s == 'F' ? 1 : 2); }

// Builds the list of clip names for one announcement. Names starting with "_" are silences
// ("_lead" before the sentence, "_gap" between two things). Returns how many names were written.
//   hasClip(name) tells whether an audio clip exists; if a thing has no clip it is called "obstacle".
static int buildPhrase(const Scene& sc, HasClipFn hasClip, char out[][NAME_LEN], int maxOut) {
  int n = 0;
  addWord(out, n, maxOut, "_lead");
  if (sc.count == 0) {
    addWord(out, n, maxOut, "clear");
    return n;
  }

  // order: left, then front, then right (nearest first inside the same side)
  int order[MAX_ITEMS];
  for (int i = 0; i < sc.count; i++) order[i] = i;
  for (int i = 1; i < sc.count; i++) {
    int k = order[i];
    int j = i - 1;
    while (j >= 0) {
      const Item& a = sc.items[order[j]];
      const Item& b = sc.items[k];
      int da = a.dm < 0 ? 1000000 : a.dm, db = b.dm < 0 ? 1000000 : b.dm;
      bool aAfterB = sideRank(a.side) > sideRank(b.side) ||
                     (sideRank(a.side) == sideRank(b.side) && da > db);
      if (!aAfterB) break;
      order[j + 1] = order[j];
      j--;
    }
    order[j + 1] = k;
  }

  for (int i = 0; i < sc.count; i++) {
    const Item& it = sc.items[order[i]];
    if (i > 0) addWord(out, n, maxOut, "_gap");
    addWord(out, n, maxOut, it.side == 'L' ? "left" : (it.side == 'R' ? "right" : "front"));
    addWord(out, n, maxOut, hasClip(it.label) ? it.label : "obstacle");
    if (it.dm >= 0 && (!it.person || SPEAK_PERSON_DISTANCE)) addDistance(out, n, maxOut, it.dm);
  }
  return n;
}
