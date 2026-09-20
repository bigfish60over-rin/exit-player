// EXIT PLAYER Service Worker
// アプリ本体（HTML・アイコン）だけを端末に置き、電波なしでも起動できるようにする。
// 曲・歌詞・GitHub API はここでは扱わない（曲はアプリ内の「端末に取り込む」で保存する）。
// index.html を更新したら、下の VERSION を上げること（例：v1 → v2）。
const VERSION = "exit-player-v1";
const SHELL = ["./", "./index.html", "./manifest.json", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(VERSION).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== VERSION).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET") return;
  // 自分のサイトのアプリ本体だけ：ネット優先、ダメなら保存済みを返す
  const isShell = url.origin === location.origin && !url.pathname.includes("/music/") && !url.pathname.endsWith("lyrics.json");
  if (!isShell) return; // 曲・歌詞・API はそのまま通す
  e.respondWith(
    fetch(e.request).then(res => {
      const copy = res.clone(); caches.open(VERSION).then(c => c.put(e.request, copy)); return res;
    }).catch(() => caches.match(e.request).then(r => r || caches.match("./index.html")))
  );
});
