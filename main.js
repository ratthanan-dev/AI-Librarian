// static/js/main.js
// "สมอง" ของ Frontend สำหรับ AI Librarian V5.4 (เวอร์ชันแก้ไขการเลื่อนหน้าจอ)

document.addEventListener('DOMContentLoaded', () => {

    // --- 1. DOM Element Selection ---
    const chatWindow = document.getElementById('chat-window');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const micBtn = document.getElementById('mic-btn');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const voiceModeBtn = document.getElementById('voice-mode-btn');
    const exitVoiceModeBtn = document.getElementById('exit-voice-mode-btn');
    const voiceModeUI = document.getElementById('voice-mode-ui');
    const voiceStatusText = document.getElementById('voice-status-text');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const muteBtn = document.getElementById('mute-btn');
    const langToggleBtn = document.getElementById('lang-toggle-btn');
    const modeToggleBtn = document.getElementById('mode-toggle-btn');
    const modeIconRag = document.getElementById('mode-icon-rag');
    const modeIconGeneral = document.getElementById('mode-icon-general');
    const modeText = document.getElementById('mode-text');


    // --- 2. State Management ---
    let currentTheme = localStorage.getItem('theme') || 'light';
    let audioPlayer = new Audio();
    let lastSpokenText = null;
    let isMuted = false;
    let isVoiceModeActive = false;
    let currentLang = localStorage.getItem('language') || 'th';
    let currentMode = 'rag';


    // --- 3. Localization (i18n) System ---
    const translations = {
        th: {
            greeting: "สวัสดีครับ, ไลท์พร้อมให้คำแนะนำแล้วครับ",
            placeholder: "พิมพ์คำถามของคุณที่นี่...",
            voice_listening: "กำลังรอรับคำสั่งเสียง...",
            exit_voice_mode: "ออกจากโหมดเสียง",
            mode_rag: "โหมดหนังสือ",
            mode_general: "โหมดคุยเล่น",
            voice_mode: "โหมดเสียง",
            play: "เล่น",
            pause: "หยุด",
            mute: "ปิดเสียง",
            unmute: "เปิดเสียง",
            toggle_theme: "เปลี่ยนธีม",
            lang_toggle: "EN",
            error_connect: "ต้องขออภัยด้วยครับ, เกิดข้อผิดพลาดในการเชื่อมต่อกับเซิร์ฟเวอร์",
            voice_error: "ไม่สามารถรับเสียงได้ กรุณาลองอีกครั้งครับ",
            voice_processing: "กำลังประมวลผล...",
            voice_error_occurred: "ขออภัยครับ เกิดข้อผิดพลาด",
            tts_generating: "กำลังสร้างเสียง..."
        },
        en: {
            greeting: "Hello, I'm Lite, ready to assist you.",
            placeholder: "Type your question here...",
            voice_listening: "Listening for voice command...",
            exit_voice_mode: "Exit Voice Mode",
            mode_rag: "Book Mode",
            mode_general: "Chat Mode",
            voice_mode: "Voice Mode",
            play: "Play",
            pause: "Pause",
            mute: "Mute",
            unmute: "Unmute",
            toggle_theme: "Toggle Theme",
            lang_toggle: "TH",
            error_connect: "Sorry, a server connection error occurred.",
            voice_error: "Could not recognize speech. Please try again.",
            voice_processing: "Processing...",
            voice_error_occurred: "Sorry, an error occurred.",
            tts_generating: "Generating audio..."
        }
    };

    const applyLanguage = (lang) => {
        document.documentElement.lang = lang;
        localStorage.setItem('language', lang);
        currentLang = lang;
        document.querySelectorAll('[data-lang-key]').forEach(element => {
            const key = element.getAttribute('data-lang-key');
            if (translations[lang][key]) {
                if (element.tagName === 'TEXTAREA' || element.tagName === 'INPUT') {
                    element.placeholder = translations[lang][key];
                } else {
                    element.textContent = translations[lang][key];
                }
            }
        });
        updateModeButtonText();
        updateMuteButtonText();
        updatePlayPauseButtonText();
    };


    // --- 4. Web Speech API (Speech-to-Text) Initialization ---
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition;
    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.lang = currentLang === 'th' ? 'th-TH' : 'en-US';
        recognition.interimResults = false;
    } else {
        console.error("Speech Recognition not supported in this browser.");
        micBtn.disabled = true;
        voiceModeBtn.disabled = true;
    }

    // --- 5. Core Functions ---

    const appendMessage = (text, sender, language = 'th') => {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) typingIndicator.parentElement.remove();

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';

        if (sender === 'ai') {
            const statusId = `tts-status-${Date.now()}`;
            const statusDiv = document.createElement('div');
            statusDiv.id = statusId;
            statusDiv.className = 'tts-status';
            statusDiv.textContent = translations[language].tts_generating;
            bubbleDiv.appendChild(statusDiv);

            const contentDiv = document.createElement('div');
            contentDiv.innerHTML = marked.parse(text);
            contentDiv.querySelectorAll('pre code').forEach(hljs.highlightElement);
            bubbleDiv.appendChild(contentDiv);
            
            lastSpokenText = { text: text, language: language, statusId: statusId };
            updateAudioButtonsVisibility(true);
        } else {
            bubbleDiv.textContent = text;
        }

        messageDiv.appendChild(bubbleDiv);
        chatWindow.appendChild(messageDiv);

        // ### <<< CHANGE START: IMPROVED SCROLLING LOGIC >>> ###
        // แทนที่ scrollToBottom() ด้วย scrollIntoView() เพื่อเลื่อนไปยัง "จุดเริ่มต้น" ของข้อความใหม่
        messageDiv.scrollIntoView({ behavior: "smooth", block: "start" });
        // ### <<< CHANGE END >>> ###
    };

    const showTypingIndicator = (show) => {
        const existingIndicator = document.querySelector('.typing-indicator');
        if (existingIndicator) existingIndicator.parentElement.remove();
        if (show) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ai-message';
            messageDiv.innerHTML = `<div class="message-bubble"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
            chatWindow.appendChild(messageDiv);
            // ยังคงใช้ scrollToBottom ที่นี่เพื่อให้ typing indicator อยู่ด้านล่างสุดเสมอ
            scrollToBottom();
        }
    };
    
    // ฟังก์ชันนี้ยังคงมีประโยชน์สำหรับ typing indicator
    const scrollToBottom = () => {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    const sendMessage = async () => {
        const query = messageInput.value.trim();
        if (!query) return;

        appendMessage(query, 'user');
        messageInput.value = '';
        messageInput.style.height = 'auto';
        showTypingIndicator(true);
        updateAudioButtonsVisibility(false);

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, mode: currentMode })
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const data = await response.json();
            appendMessage(data.answer, 'ai', data.language);
            playLastResponse();

        } catch (error) {
            console.error("Error sending message:", error);
            appendMessage(translations[currentLang].error_connect, 'ai', currentLang);
        }
    };

    // --- 6. Theme Management ---
    const applyTheme = (theme) => {
        document.documentElement.className = theme;
        localStorage.setItem('theme', theme);
        currentTheme = theme;
    };
    const toggleTheme = () => applyTheme(currentTheme === 'light' ? 'dark' : 'light');

    // --- 7. Audio Player & TTS ---
    const updateAudioButtonsVisibility = (show) => {
        playPauseBtn.classList.toggle('hidden', !show || !lastSpokenText);
        muteBtn.classList.toggle('hidden', !show || !lastSpokenText);
    };

    const playLastResponse = async () => {
        if (!lastSpokenText || !lastSpokenText.text) return;
        
        const statusElement = document.getElementById(lastSpokenText.statusId);

        try {
            const response = await fetch('/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: lastSpokenText.text, 
                    language: lastSpokenText.language 
                })
            });
            const data = await response.json();
            if (data.audio_url) {
                audioPlayer.src = data.audio_url;
                audioPlayer.muted = isMuted;
                audioPlayer.play();
            }
        } catch (error) {
            console.error("Error fetching TTS:", error);
        } finally {
            if (statusElement) {
                statusElement.remove();
            }
        }
    };

    const togglePlayPause = () => {
        if (audioPlayer.paused) {
            audioPlayer.src ? audioPlayer.play() : playLastResponse();
        } else {
            audioPlayer.pause();
        }
    };
    
    const updatePlayPauseButtonText = () => {
        playPauseBtn.textContent = audioPlayer.paused ? translations[currentLang].play : translations[currentLang].pause;
    };
    const updateMuteButtonText = () => {
        muteBtn.textContent = isMuted ? translations[currentLang].unmute : translations[currentLang].mute;
    };
    const toggleMute = () => {
        isMuted = !isMuted;
        audioPlayer.muted = isMuted;
        updateMuteButtonText();
    };


    // --- 8. Voice Mode & Speech Recognition Logic ---
    const startSpeechRecognition = () => {
        if (recognition && isVoiceModeActive) {
            recognition.lang = currentLang === 'th' ? 'th-TH' : 'en-US';
            try {
                recognition.start();
                console.log("Speech recognition started.");
            } catch(e) {
                console.error("Error starting recognition (might already be active):", e);
            }
        }
    };

    const handleVoiceMode = async (query) => {
        voiceStatusText.textContent = translations[currentLang].voice_processing;
        try {
            const response = await fetch('/voice_mode_ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: query, 
                    mode: currentMode
                })
            });
            const data = await response.json();
            
            voiceStatusText.textContent = data.answer;
            
            if (data.audio_url) {
                audioPlayer.src = data.audio_url;
                audioPlayer.muted = isMuted;
                audioPlayer.play();
            } else {
                setTimeout(() => {
                    if (isVoiceModeActive) {
                        voiceStatusText.textContent = translations[currentLang].voice_listening;
                        startSpeechRecognition();
                    }
                }, 1000);
            }

        } catch (error) {
            console.error("Error in voice mode:", error);
            voiceStatusText.textContent = translations[currentLang].voice_error_occurred;
            setTimeout(() => {
                if (isVoiceModeActive) {
                    voiceStatusText.textContent = translations[currentLang].voice_listening;
                    startSpeechRecognition();
                }
            }, 2000);
        }
    };

    // --- 9. Button Logic ---
    const updateModeButtonText = () => {
        modeText.textContent = currentMode === 'rag' ? translations[currentLang].mode_rag : translations[currentLang].mode_general;
    };

    modeToggleBtn.addEventListener('click', () => {
        currentMode = currentMode === 'rag' ? 'general' : 'rag';
        modeIconRag.classList.toggle('hidden', currentMode !== 'rag');
        modeIconGeneral.classList.toggle('hidden', currentMode !== 'general');
        updateModeButtonText();
    });

    langToggleBtn.addEventListener('click', () => {
        const newLang = currentLang === 'th' ? 'en' : 'th';
        applyLanguage(newLang);
    });

    // --- 10. Event Listeners ---
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = `${messageInput.scrollHeight}px`;
    });

    themeToggleBtn.addEventListener('click', toggleTheme);
    playPauseBtn.addEventListener('click', togglePlayPause);
    muteBtn.addEventListener('click', toggleMute);
    audioPlayer.addEventListener('play', updatePlayPauseButtonText);
    audioPlayer.addEventListener('pause', updatePlayPauseButtonText);
    
    audioPlayer.addEventListener('ended', () => {
        updatePlayPauseButtonText();
        if (isVoiceModeActive) {
            console.log("Voice mode audio ended. Restarting recognition.");
            voiceStatusText.textContent = translations[currentLang].voice_listening;
            startSpeechRecognition();
        }
    });

    micBtn.addEventListener('click', () => {
        micBtn.disabled = true;
        startSpeechRecognition();
    });

    voiceModeBtn.addEventListener('click', () => {
        isVoiceModeActive = true;
        voiceModeUI.classList.remove('hidden');
        chatWindow.classList.add('hidden');
        document.getElementById('input-area').classList.add('hidden');
        document.getElementById('toolbar').classList.add('hidden');
        voiceStatusText.textContent = translations[currentLang].voice_listening;
        startSpeechRecognition();
    });

    exitVoiceModeBtn.addEventListener('click', () => {
        isVoiceModeActive = false;
        if (recognition) recognition.stop();
        audioPlayer.pause();
        voiceModeUI.classList.add('hidden');
        chatWindow.classList.remove('hidden');
        document.getElementById('input-area').classList.remove('hidden');
        document.getElementById('toolbar').classList.remove('hidden');
    });

    if (recognition) {
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (isVoiceModeActive) {
                handleVoiceMode(transcript);
            } else {
                messageInput.value = transcript;
                sendMessage();
            }
        };

        recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
            if (isVoiceModeActive) {
                voiceStatusText.textContent = translations[currentLang].voice_error;
                 setTimeout(() => {
                    if (isVoiceModeActive) {
                        voiceStatusText.textContent = translations[currentLang].voice_listening;
                        startSpeechRecognition();
                    }
                }, 2000);
            }
            micBtn.disabled = false;
        };
        
        recognition.onend = () => {
            console.log("Speech recognition service ended.");
            if (!isVoiceModeActive) {
                micBtn.disabled = false;
            }
        };
    }

    // --- 11. Initial Setup ---
    applyTheme(currentTheme);
    applyLanguage(currentLang);
});

