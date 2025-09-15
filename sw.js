// static/sw.js
// Service Worker พื้นฐานสำหรับทำให้แอปสามารถติดตั้งได้ (Installable)

self.addEventListener('fetch', (event) => {
  // ในตอนนี้เรายังไม่ทำระบบ offline caching
  // แค่มีไฟล์นี้ไว้เพื่อให้เบราว์เซอร์รู้จักแอปของเราในฐานะ PWA ก็เพียงพอแล้ว
});
