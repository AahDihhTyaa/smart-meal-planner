// app.js — 3D Glassmorphic Bento Grid Frontend Controller

let activeIngredients = [];
let dayPlan = {};
let weekPlan = {};
let viewMode = 'daily'; // 'daily' or 'weekly'
let selectedWeeklyDay = 'Monday';
let chatHistory = [];
let isVoiceRecording = false;
let recognitionInstance = null;
let currentSwappedIngredients = {}; // Maps original -> swapped
let alternativeRecipes = [];
let unlockableRecipes = [];

document.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    initDragAndDrop();
    initSpeechRecognition();
    
function loadSettings() {
    const indianFocus = localStorage.getItem('indianFocus') === 'true';
    const focusEl = document.getElementById('indian-focus');
    if (focusEl) {
        focusEl.checked = indianFocus;
        focusEl.addEventListener('change', (e) => {
            localStorage.setItem('indianFocus', e.target.checked);
        });
    }
}

// Check if there are saved ingredients or load empty state
    updatePantryUI();
});

function getAuthHeaders() {
    return {
        'Content-Type': 'application/json'
    };
}

// --- Drag & Drop Image Upload (Module 1) ---
function initDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('vision-upload');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-cyan-500', 'bg-cyan-500/[0.05]');
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('border-cyan-500', 'bg-cyan-500/[0.05]');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            handleImageFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleImageFile(e.target.files[0]);
        }
    });
}

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast("Please upload an image file.", "error");
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('image-preview');
        const dropIcon = document.getElementById('drop-icon');
        const dropText = document.getElementById('drop-text');
        
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        dropIcon.classList.add('hidden');
        dropText.classList.add('hidden');

        document.getElementById('process-photo-btn').disabled = false;
    };
    reader.readAsDataURL(file);
}

async function processPhoto() {
    const preview = document.getElementById('image-preview');
    if (!preview.src) return;

    showLoader(true, "Scanning ingredients with AI Vision...");

    try {
        const base64Data = preview.src.split(',')[1];
        const mimeType = preview.src.split(';')[0].split(':')[1] || 'image/jpeg';
        const response = await fetch('/api/vision', {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({ 
                image_base64: base64Data,
                mime_type: mimeType
            })
        });
        
        const data = await response.json();
        if (data.success && data.ingredients) {
            data.ingredients.forEach(item => {
                let name = "";
                if (typeof item === 'string') {
                    name = item.toLowerCase();
                } else if (item && item.name) {
                    const qty = item.quantity ? item.quantity.trim() : "";
                    const rawName = item.name.trim();
                    if (qty) {
                        name = `${qty} ${rawName}`.toLowerCase();
                    } else {
                        name = rawName.toLowerCase();
                    }
                }
                if (name && !activeIngredients.includes(name)) {
                    activeIngredients.push(name);
                }
            });
            updatePantryUI();
            showToast(`Detected ${data.ingredients.length} items!`, "success");
            
            // Clear photo zone
            resetDropZone();

            // Auto-trigger plan generation
            if (activeIngredients.length > 0) {
                generatePlan();
            }
        } else {
            showToast(data.error || "Vision analysis failed.", "error");
        }
    } catch (err) {
        console.error("Vision Error:", err);
        showToast("Error processing vision request: " + err.message, "error");
    } finally {
        showLoader(false);
    }
}

function resetDropZone() {
    const preview = document.getElementById('image-preview');
    const dropIcon = document.getElementById('drop-icon');
    const dropText = document.getElementById('drop-text');
    
    preview.src = "";
    preview.classList.add('hidden');
    dropIcon.classList.remove('hidden');
    dropText.classList.remove('hidden');
    document.getElementById('process-photo-btn').disabled = true;
    document.getElementById('vision-upload').value = "";
}

// --- Browser Speech Recognition (Module 1) ---
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    recognitionInstance = new SpeechRecognition();
    recognitionInstance.lang = 'en-US';
    recognitionInstance.continuous = false;
    recognitionInstance.interimResults = false;

    recognitionInstance.onstart = () => {
        isVoiceRecording = true;
        const btn = document.getElementById('mic-toggle-btn');
        btn.classList.add('mic-pulse', 'bg-cyan-500/20', 'border-cyan-500');
        showToast("Listening to ingredients...", "info");
    };

    recognitionInstance.onresult = (event) => {
        const text = event.results[0][0].transcript;
        const input = document.getElementById('raw-ingredients-input');
        input.value = text;
        showToast("Voice captured!", "success");
        addRawIngredients();
    };

    recognitionInstance.onerror = (e) => {
        console.error("STT Error", e.error);
        showToast("Speech recognition error: " + e.error, "error");
        stopVoiceRecording();
    };

    recognitionInstance.onend = () => {
        stopVoiceRecording();
    };
}

function toggleVoiceInput() {
    if (!recognitionInstance) {
        showToast("Voice Recognition not supported in this browser. Use Chrome/Edge.", "error");
        return;
    }
    if (isVoiceRecording) {
        recognitionInstance.stop();
    } else {
        recognitionInstance.start();
    }
}

function stopVoiceRecording() {
    isVoiceRecording = false;
    const btn = document.getElementById('mic-toggle-btn');
    btn.classList.remove('mic-pulse', 'bg-cyan-500/20', 'border-cyan-500');
}

// --- Pantry Management (Module 2) ---
function addRawIngredients() {
    const input = document.getElementById('raw-ingredients-input');
    const text = input.value.trim();
    if (!text) return;

    // Split by commas/newlines and normalize
    const items = text.split(/[,\n]+/).map(item => item.trim().toLowerCase()).filter(item => item.length > 0);
    
    items.forEach(item => {
        if (!activeIngredients.includes(item)) {
            activeIngredients.push(item);
        }
    });

    input.value = "";
    updatePantryUI();
    showToast(`Added ${items.length} raw text item(s)`, "success");
}

function removePantryIngredient(ing) {
    activeIngredients = activeIngredients.filter(item => item !== ing);
    updatePantryUI();
}

function clearPantry() {
    activeIngredients = [];
    updatePantryUI();
    showToast("Pantry cleared", "info");
}

function updatePantryUI() {
    const container = document.getElementById('pantry-chips-container');
    container.innerHTML = "";

    if (activeIngredients.length === 0) {
        container.innerHTML = `<div class="text-xs text-white/30 italic m-auto">Pantry is empty. Upload a photo, speak, or type items.</div>`;
        return;
    }

    activeIngredients.forEach(ing => {
        const chip = document.createElement('div');
        chip.className = "chip flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-white/5 border border-white/10 hover:border-white/20 transition-all text-white/90";
        chip.innerHTML = `
            <span>${ing}</span>
            <button onclick="removePantryIngredient('${ing}')" class="text-[10px] text-white/40 hover:text-white font-bold ml-1">✕</button>
        `;
        container.appendChild(chip);
    });
}

// --- Plan & Scheduler Generation (Module 3) ---
async function generatePlan() {
    if (activeIngredients.length === 0) {
        showToast("Please add at least one ingredient to your pantry first.", "warning");
        return;
    }

    showLoader(true, "AI Chef is designing daily and weekly menus...");

    const filters = []; // dietary filters could be loaded from settings
    const indianFocus = document.getElementById('indian-focus').checked;

    try {
        const response = await fetch('/api/plan', {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({
                ingredients: activeIngredients.join(', '),
                dietary_filters: filters,
                indian_focus: indianFocus
            })
        });

        const data = await response.json();
        if (data.success) {
            dayPlan = data.day_plan || {};
            weekPlan = data.week_plan || {};
            alternativeRecipes = data.alternatives || [];
            unlockableRecipes = data.unlockable || [];
            currentSwappedIngredients = {}; // reset swaps
            
            // Re-render UI
            renderRecipePlan();
            renderAlternativeRecipes();
            renderUnlockableRecipes();
            
            // Get Analytics
            await fetchAnalytics();

            showToast("Plan generated successfully!", "success");
        } else {
            showToast(data.error || "Failed to generate meal plan.", "error");
        }
    } catch (err) {
        console.error(err);
        showToast("Error connecting to server plan API.", "error");
    } finally {
        showLoader(false);
    }
}

function switchViewMode(mode) {
    viewMode = mode;
    document.getElementById('tab-daily').className = mode === 'daily' ? 
        "px-4 py-1.5 text-xs font-semibold rounded-lg text-white bg-amber-600 transition" : 
        "px-4 py-1.5 text-xs font-semibold rounded-lg text-white/55 hover:text-white transition";
    
    document.getElementById('tab-weekly').className = mode === 'weekly' ? 
        "px-4 py-1.5 text-xs font-semibold rounded-lg text-white bg-amber-600 transition" : 
        "px-4 py-1.5 text-xs font-semibold rounded-lg text-white/55 hover:text-white transition";
        
    const dayTabs = document.getElementById('weekly-days-tabs');
    if (mode === 'weekly') {
        dayTabs.classList.remove('hidden');
        renderWeeklyDaysTabs();
    } else {
        dayTabs.classList.add('hidden');
    }

    renderRecipePlan();
}

function renderWeeklyDaysTabs() {
    const container = document.getElementById('weekly-days-tabs');
    container.innerHTML = "";
    
    const days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
    days.forEach(day => {
        const btn = document.createElement('button');
        btn.onclick = () => {
            selectedWeeklyDay = day;
            renderWeeklyDaysTabs();
            renderRecipePlan();
        };
        btn.textContent = day.substring(0,3);
        btn.title = day;
        btn.className = selectedWeeklyDay === day ?
            "px-3 py-1 text-[11px] font-bold rounded-lg bg-white/10 text-amber-400 border border-amber-500/30 shrink-0" :
            "px-3 py-1 text-[11px] font-medium rounded-lg text-white/50 hover:text-white shrink-0";
        container.appendChild(btn);
    });
}

function renderRecipePlan() {
    const container = document.getElementById('recipe-list-container');
    container.innerHTML = "";

    const activePlan = (viewMode === 'daily') ? dayPlan : (weekPlan[selectedWeeklyDay] || {});
    const slots = ["breakfast", "lunch", "dinner"];
    
    let hasMeals = false;

    slots.forEach(slot => {
        let recipe = activePlan[slot];
        if (recipe) {
            hasMeals = true;
            
            // Compute ingredient coverage %
            let coverage = Math.round((recipe.coverage || 0) * 100);
            
            // Check if there is an active swap for cheaper options
            const matchedIngredients = Array.isArray(recipe.ingredient_list) ? recipe.ingredient_list : [];
            let swappedIngredients = matchedIngredients.map(ing => {
                const clean = ing.replace(/_/g, ' ');
                if (currentSwappedIngredients[ing]) {
                    return `<span class="text-emerald-400 font-semibold">${currentSwappedIngredients[ing]}</span>`;
                }
                return clean;
            });

            const card = document.createElement('div');
            card.className = "bento-card rounded-2xl p-4 flex gap-4 hover:border-amber-500/40 cursor-pointer relative group";
            card.onclick = (e) => {
                if (e.target.closest('.refresh-btn')) return;
                openRecipeModal(recipe, slot);
            };
            
            // Radial Progress Circle logic
            const dashArray = 125.6;
            const dashOffset = dashArray - (dashArray * coverage / 100);

            card.innerHTML = `
                <!-- Coverage Circular Ring -->
                <div class="flex flex-col items-center justify-center shrink-0">
                    <div class="relative w-14 h-14 flex items-center justify-center">
                        <svg class="w-full h-full transform -rotate-90">
                            <circle cx="28" cy="28" r="22" stroke="rgba(255,255,255,0.04)" stroke-width="3.5" fill="transparent"/>
                            <circle cx="28" cy="28" r="22" stroke="url(#amber-grad)" stroke-width="3.5" fill="transparent"
                                stroke-dasharray="${dashArray}" stroke-dashoffset="${dashOffset}" class="progress-circle shadow-[0_0_8px_rgba(245,158,11,0.5)]"/>
                            <defs>
                                <linearGradient id="amber-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stop-color="#F59E0B" />
                                    <stop offset="100%" stop-color="#D97706" />
                                </linearGradient>
                            </defs>
                        </svg>
                        <span class="absolute text-[10px] font-mono font-bold">${coverage}%</span>
                    </div>
                    <span class="text-[9px] text-white/40 uppercase font-semibold mt-1">Match</span>
                </div>

                <!-- Info -->
                <div class="flex-1 flex flex-col justify-between">
                    <div>
                        <div class="flex justify-between items-center">
                            <span class="text-[10px] font-bold tracking-wider uppercase text-amber-500 bg-amber-500/10 px-2 py-0.5 rounded-full">${slot}</span>
                            <div class="flex items-center gap-2">
                                <span class="text-[10px] font-semibold text-white/40">${recipe.prep_time_min || 20}m | Diff: ${recipe.difficulty || 2}/5</span>
                                <button onclick="refreshRecipeSlot('${slot}')" class="refresh-btn text-white/50 hover:text-white transition p-1 hover:bg-white/10 rounded-full" title="Swap Recipe">
                                    <span class="text-xs">🔄</span>
                                </button>
                            </div>
                        </div>
                        <h3 class="serif-title text-md font-semibold text-white mt-1.5 leading-snug">${recipe.title}</h3>
                        <p class="text-xs text-white/60 line-clamp-2 mt-1">${recipe.instructions || ''}</p>
                    </div>
                    
                    <div class="text-[10px] text-white/40 mt-2 truncate">
                        <strong>Includes:</strong> ${swappedIngredients.join(', ')}
                    </div>
                </div>
            `;
            container.appendChild(card);
        }
    });

    if (!hasMeals) {
        container.innerHTML = `
            <div class="m-auto text-center py-12 px-4 flex flex-col items-center gap-3">
                <span class="text-5xl animate-bounce">🥗</span>
                <h3 class="serif-title text-lg font-semibold text-white/70">Your kitchen is quiet</h3>
                <p class="text-xs text-white/40 max-w-xs">Fill up the normalizer on the left and tap "Generate" to unlock optimized custom recipe recommendations.</p>
            </div>
        `;
    }

    // Update aggregate nutrition/budget statistics
    updateDashboardStats();
}

function renderAlternativeRecipes() {
    const container = document.getElementById('alternative-recipes-container');
    container.innerHTML = "";

    if (alternativeRecipes.length === 0) {
        container.innerHTML = `<div class="col-span-2 text-center text-xs text-white/30 italic py-4">No alternative recommendations loaded.</div>`;
        return;
    }

    alternativeRecipes.forEach(recipe => {
        let coverage = Math.round((recipe.coverage || 0) * 100);
        
        const card = document.createElement('div');
        card.className = "mini-card rounded-xl p-3 flex gap-2.5 hover:border-amber-500/30 cursor-pointer transition";
        card.onclick = () => openRecipeModal(recipe, 'alternative');
        
        card.innerHTML = `
            <!-- Coverage Percent -->
            <div class="flex flex-col items-center justify-center shrink-0">
                <span class="text-xs font-mono font-bold text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded">${coverage}%</span>
            </div>
            
            <div class="flex-1 min-w-0">
                <h4 class="text-xs font-semibold text-white truncate leading-tight">${recipe.title}</h4>
                <div class="flex gap-2 text-[9px] text-white/40 mt-1">
                    <span>${recipe.calories || 0} kcal</span>
                    <span>•</span>
                    <span>${recipe.prep_time_min || 20}m</span>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

// --- Analytics, Budget & Nutrition (Module 4) ---
async function fetchAnalytics() {
    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({
                ingredients: activeIngredients,
                day_plan: dayPlan,
                week_plan: weekPlan
            })
        });

        const data = await response.json();
        if (data.success && data.analysis) {
            renderAnalyticalInsights(data.analysis);
        }
    } catch (err) {
        console.error("Analytics fetch err", err);
    }
}

function renderAnalyticalInsights(analysis) {
    const container = document.getElementById('analytical-insight');
    container.innerHTML = "";

    const confidence = analysis.confidence || {};
    const staples = analysis.staple_unlocks || [];
    const batch = analysis.batch_cooking || [];
    const swaps = analysis.seasonal_swaps || [];

    let detailsHTML = `
        <div class="flex flex-col gap-2.5 text-xs text-white/60">
            <div class="flex justify-between items-center bg-white/5 p-2 rounded-xl border border-white/5">
                <span>Pantry Health Grade:</span>
                <strong class="text-emerald-400 text-sm font-bold">${confidence.grade || 'C'} (${confidence.score || 50} pts)</strong>
            </div>
    `;

    if (staples.length > 0) {
        detailsHTML += `
            <div>
                <span class="text-white/35 font-bold uppercase text-[10px] block mb-1">Staple Unlocks:</span>
                <p>Adding <strong class="text-amber-400">${staples[0].ingredient}</strong> unlocks <strong>${staples[0].unlocks}</strong> new recipes like <em>${staples[0].top_recipes.slice(0, 2).join(', ')}</em>.</p>
            </div>
        `;
    }

    if (batch.length > 0) {
        detailsHTML += `
            <div class="mt-1">
                <span class="text-white/35 font-bold uppercase text-[10px] block mb-1">Batch Cook Alert:</span>
                <p>Prepare <strong class="text-cyan-400">${batch[0].ingredient}</strong> in advance. It is used in <strong>${batch[0].used_in}</strong> meals this cycle.</p>
            </div>
        `;
    }

    if (swaps.length > 0) {
        detailsHTML += `
            <div class="mt-1">
                <span class="text-white/35 font-bold uppercase text-[10px] block mb-1">Seasonal Swap:</span>
                <p>Swap out <strong class="text-rose-400">${swaps[0].original}</strong> with <strong class="text-emerald-400">${swaps[0].swap}</strong> for peak freshness.</p>
            </div>
        `;
    }

    detailsHTML += `</div>`;
    container.innerHTML = detailsHTML;
}

function updateDashboardStats() {
    const activePlan = (viewMode === 'daily') ? dayPlan : (weekPlan[selectedWeeklyDay] || {});
    const slots = ["breakfast", "lunch", "dinner"];
    
    let totalCals = 0;
    let totalProtein = 0;
    let totalCarbs = 0;
    let totalFat = 0;
    let totalCost = 0.00;

    slots.forEach(slot => {
        const recipe = activePlan[slot];
        if (recipe) {
            totalCals += parseInt(recipe.calories) || 0;
            totalProtein += parseFloat(recipe.protein) || 0;
            totalCarbs += parseFloat(recipe.carbs) || 0;
            totalFat += parseFloat(recipe.fat) || 0;

            // Compute custom base price per recipe
            totalCost += getEstimatedRecipeCost(recipe);
        }
    });

    // Update Nutrition Numbers
    document.getElementById('nutri-cals').textContent = `${totalCals} kcal`;
    document.getElementById('nutri-protein').textContent = `${Math.round(totalProtein)}g`;
    document.getElementById('nutri-carbs').textContent = `${Math.round(totalCarbs)}g`;
    document.getElementById('nutri-fat').textContent = `${Math.round(totalFat)}g`;

    // Max ceilings for percentages
    const maxCals = 2500;
    const maxProtein = 150;
    const maxCarbs = 300;
    const maxFat = 80;

    document.getElementById('nutri-cals-bar').style.width = `${Math.min(100, (totalCals / maxCals) * 100)}%`;
    document.getElementById('nutri-protein-bar').style.width = `${Math.min(100, (totalProtein / maxProtein) * 100)}%`;
    document.getElementById('nutri-carbs-bar').style.width = `${Math.min(100, (totalCarbs / maxCarbs) * 100)}%`;
    document.getElementById('nutri-fat-bar').style.width = `${Math.min(100, (totalFat / maxFat) * 100)}%`;

    // Budget Costs Calculation
    const budgetLimit = 15.00;
    document.getElementById('budget-ratio').textContent = `$${totalCost.toFixed(2)} / $${budgetLimit.toFixed(2)}`;
    document.getElementById('budget-progress').style.width = `${Math.min(100, (totalCost / budgetLimit) * 100)}%`;
}

function getEstimatedRecipeCost(recipe) {
    const matchedIngredients = Array.isArray(recipe.ingredient_list) ? recipe.ingredient_list : [];
    
    // Static basic pricing lookup database
    const prices = {
        "salmon": 3.50, "tuna": 2.80, "shrimp": 3.20, "beef": 3.00, "lamb": 3.50,
        "chicken": 2.00, "chicken_breast": 2.20, "tofu": 1.00, "paneer": 1.50,
        "egg": 0.30, "eggs": 0.30, "milk": 0.25, "cheese": 0.80, "butter": 0.30,
        "rice": 0.20, "pasta": 0.25, "tomato": 0.40, "onion": 0.20, "garlic": 0.10
    };

    let cost = 0.00;
    matchedIngredients.forEach(ing => {
        const base = ing.toLowerCase();
        // Check for active cheap swap
        if (currentSwappedIngredients[ing]) {
            cost += 0.80; // generic cheap alternative cost
        } else {
            cost += prices[base] || 0.35; // default fallback
        }
    });

    return cost > 0 ? cost : 2.50; // default recipe base
}

function toggleSmartSwaps(checkbox) {
    const enabled = checkbox.checked;
    
    if (enabled) {
        // Find expensive ingredients in current plan and assign swap alternatives
        const expensiveMapping = {
            "salmon": "tilapia",
            "beef": "chicken",
            "lamb": "pork",
            "shrimp": "tofu",
            "prawn": "tofu",
            "steak": "chicken breast",
            "parmesan": "cheddar cheese"
        };

        const activePlan = (viewMode === 'daily') ? dayPlan : (weekPlan[selectedWeeklyDay] || {});
        const slots = ["breakfast", "lunch", "dinner"];
        
        let swappedCount = 0;
        slots.forEach(slot => {
            const recipe = activePlan[slot];
            if (recipe) {
                const matchedIngredients = Array.isArray(recipe.ingredient_list) ? recipe.ingredient_list : [];
                matchedIngredients.forEach(ing => {
                    if (expensiveMapping[ing]) {
                        currentSwappedIngredients[ing] = expensiveMapping[ing];
                        swappedCount++;
                    }
                });
            }
        });

        if (swappedCount > 0) {
            showToast(`Smart Swaps: Replaced ${swappedCount} expensive items!`, "success");
        } else {
            showToast("No expensive ingredients to swap in current plan.", "info");
        }
    } else {
        currentSwappedIngredients = {};
        showToast("Swaps disabled.", "info");
    }

    renderRecipePlan();
}

// --- Chatbot Terminal (Module 5) ---
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;

    // Append user message
    appendMessage('user', msg);
    input.value = "";

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({
                message: msg,
                context: chatHistory.map(h => `${h.role}: ${h.text}`)
            })
        });

        const data = await response.json();
        if (data.success && data.response) {
            appendMessage('bot', data.response);
            
            // Update Intent Badge & Border Glow
            updateChatIntent(data.intent || 'general');
        } else {
            appendMessage('bot', "Sorry, I encountered an issue processing that query.");
        }
    } catch (err) {
        appendMessage('bot', "Connection failed. Please check your configurations.");
    }
}

function handleChatKey(e) {
    if (e.key === 'Enter') {
        sendChatMessage();
    }
}

function appendMessage(role, text) {
    const container = document.getElementById('chat-messages');
    
    // Save to history
    chatHistory.push({ role, text });
    if (chatHistory.length > 20) chatHistory.shift();

    const bubble = document.createElement('div');
    if (role === 'user') {
        bubble.className = "bg-cyan-600 text-white p-3 rounded-2xl rounded-br-none self-end max-w-[90%] leading-relaxed shadow-md animate-fade-in";
        bubble.textContent = text;
    } else {
        bubble.className = "bg-white/5 text-white/80 p-3 rounded-2xl rounded-bl-none self-start max-w-[90%] leading-relaxed border border-white/5 animate-fade-in";
        bubble.innerHTML = `
            <div>${text}</div>
            <button onclick="speakText('${text.replace(/'/g, "\\'")}')" class="text-[9px] text-cyan-400 hover:underline mt-1.5 flex items-center gap-1 font-semibold uppercase">🔊 Read Aloud</button>
        `;
    }
    
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
}

function updateChatIntent(intent) {
    const badge = document.getElementById('chat-intent');
    const terminal = document.getElementById('chat-terminal');

    badge.textContent = intent;

    // Reset intent borders
    terminal.classList.remove('intent-border-cyan', 'intent-border-amber', 'intent-border-mint');

    if (intent.includes('recipe') || intent.includes('cooking')) {
        terminal.classList.add('intent-border-amber');
        badge.className = "text-[9px] font-mono uppercase bg-amber-500/10 px-2 py-0.5 rounded text-amber-400 border border-amber-500/20";
    } else if (intent.includes('diet') || intent.includes('health') || intent.includes('nutri')) {
        terminal.classList.add('intent-border-mint');
        badge.className = "text-[9px] font-mono uppercase bg-emerald-500/10 px-2 py-0.5 rounded text-emerald-400 border border-emerald-500/20";
    } else {
        terminal.classList.add('intent-border-cyan');
        badge.className = "text-[9px] font-mono uppercase bg-cyan-500/10 px-2 py-0.5 rounded text-cyan-400 border border-cyan-500/20";
    }
}

// --- Text-to-Speech (TTS) ---
function speakText(text) {
    if (!window.speechSynthesis) {
        showToast("Speech synthesis not supported in this browser.", "error");
        return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
}

// --- Recipe Detail Modal ---
let activeModalRecipe = null;
function openRecipeModal(recipe, slot) {
    activeModalRecipe = recipe;
    
    document.getElementById('modal-recipe-slot').textContent = slot;
    document.getElementById('modal-recipe-title').textContent = recipe.title;
    document.getElementById('modal-cals').textContent = `${recipe.calories || 0} kcal`;
    document.getElementById('modal-protein').textContent = `${Math.round(recipe.protein || 0)}g`;
    document.getElementById('modal-carbs').textContent = `${Math.round(recipe.carbs || 0)}g`;
    document.getElementById('modal-fat').textContent = `${Math.round(recipe.fat || 0)}g`;

    // Ingredients mapping
    const ingContainer = document.getElementById('modal-ingredients');
    ingContainer.innerHTML = "";
    
    const ingredients = Array.isArray(recipe.ingredient_list) ? recipe.ingredient_list : [];
    ingredients.forEach(ing => {
        const li = document.createElement('li');
        li.textContent = ing.replace(/_/g, ' ');
        ingContainer.appendChild(li);
    });

    // Instructions mapping
    const instContainer = document.getElementById('modal-instructions');
    instContainer.innerHTML = "";
    
    const steps = (recipe.instructions || "").split(/\d+\.\s+/).filter(s => s.trim().length > 0);
    if (steps.length > 0) {
        steps.forEach((step, idx) => {
            const p = document.createElement('p');
            p.className = "mb-2";
            p.innerHTML = `<strong class="text-amber-500">${idx + 1}.</strong> ${step.trim()}`;
            instContainer.appendChild(p);
        });
    } else {
        instContainer.textContent = recipe.instructions || "No preparation guidelines listed.";
    }

    const swapActions = document.getElementById('modal-swap-actions');
    if (slot === 'alternative') {
        swapActions.classList.remove('hidden');
    } else {
        swapActions.classList.add('hidden');
    }

    const modal = document.getElementById('recipe-modal');
    modal.classList.remove('invisible');
    modal.classList.add('opacity-100');
}

function closeRecipeModal() {
    const modal = document.getElementById('recipe-modal');
    modal.classList.remove('opacity-100');
    modal.classList.add('invisible');
    activeModalRecipe = null;
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
}

function swapActiveRecipe(targetSlot) {
    if (!activeModalRecipe) return;
    
    const activePlan = (viewMode === 'daily') ? dayPlan : (weekPlan[selectedWeeklyDay] || {});
    const previousRecipe = activePlan[targetSlot];
    
    // Set active recipe in schedule
    activePlan[targetSlot] = activeModalRecipe;
    
    // Remove active recipe from alternatives pool, and put the previous recipe back (if it existed)
    alternativeRecipes = alternativeRecipes.filter(r => r.title !== activeModalRecipe.title);
    if (previousRecipe) {
        alternativeRecipes.push(previousRecipe);
    }
    
    // Re-render schedule and alternatives
    renderRecipePlan();
    renderAlternativeRecipes();
            renderUnlockableRecipes();
    
    // Show toast and close modal
    showToast(`Successfully swapped recipe into ${targetSlot}!`, "success");
    closeRecipeModal();
}

window.refreshRecipeSlot = function(slot) {
    if (!alternativeRecipes || alternativeRecipes.length === 0) {
        showToast("No alternatives available. Generate more!", "warning");
        return;
    }
    const activePlan = (viewMode === 'daily') ? dayPlan : (weekPlan[selectedWeeklyDay] || {});
    const previousRecipe = activePlan[slot];
    
    // Pick a random alternative
    const altIndex = Math.floor(Math.random() * alternativeRecipes.length);
    const newRecipe = alternativeRecipes[altIndex];
    
    // Swap
    activePlan[slot] = newRecipe;
    alternativeRecipes.splice(altIndex, 1);
    if (previousRecipe) {
        alternativeRecipes.push(previousRecipe);
    }
    
    renderRecipePlan();
    renderAlternativeRecipes();
    showToast(`Refreshed ${slot}!`, "success");
}

function readRecipeText() {
    if (!activeModalRecipe) return;
    const text = `${activeModalRecipe.title}. Ingredients involved: ${activeModalRecipe.ingredients.replace(/\|/g, ', ')}. Instructions: ${activeModalRecipe.instructions}`;
    speakText(text);
}

// --- Utilities ---
function triggerSkeletonLoaders() {
    const recipeContainer = document.getElementById('recipe-list-container');
    if (recipeContainer) {
        recipeContainer.innerHTML = `
            <div class="flex flex-col gap-4 w-full h-full p-2">
                <div class="skeleton-shimmer w-full h-24"></div>
                <div class="skeleton-shimmer w-full h-24"></div>
                <div class="skeleton-shimmer w-full h-24"></div>
            </div>
        `;
    }
    
    const bars = ['cals', 'protein', 'carbs', 'fat'];
    bars.forEach(b => {
        const bar = document.getElementById(`nutri-${b}-bar`);
        if (bar) bar.style.width = '0%';
    });
}

let loaderTimeout;
let messageInterval;
function showLoader(show, message = "") {
    const loader = document.getElementById('loading-overlay');
    const msg = document.getElementById('loading-message');
    
    if (loaderTimeout) {
        clearTimeout(loaderTimeout);
        loaderTimeout = null;
    }
    if (messageInterval) {
        clearInterval(messageInterval);
        messageInterval = null;
    }
    
    if (show) {
        triggerSkeletonLoaders();
        msg.textContent = message || "Cooking up choices...";
        
        if (message.includes("AI Chef")) {
            const aiMessages = [
                "Consulting the Michelin-star Chef...",
                "Selecting the freshest ingredients...",
                "Designing a balanced diet...",
                "Drafting your personalized recipes...",
                "Almost ready, plating your meals..."
            ];
            let msgIdx = 0;
            messageInterval = setInterval(() => {
                msg.textContent = aiMessages[msgIdx % aiMessages.length];
                msgIdx++;
            }, 2500);
        }
        
        loader.classList.remove('invisible', 'pointer-events-none');
        // Small delay to ensure display state triggers CSS transition
        setTimeout(() => loader.classList.add('opacity-100'), 10);
    } else {
        loader.classList.remove('opacity-100');
        loader.classList.add('pointer-events-none');
        loaderTimeout = setTimeout(() => {
            loader.classList.add('invisible');
        }, 500);
    }
}

function showToast(message, type = "info") {
    // Custom Toast Alert
    const toast = document.createElement('div');
    toast.className = `fixed bottom-5 right-5 z-[9999] px-4 py-3 rounded-xl border shadow-lg text-xs font-semibold flex items-center gap-2 transform translate-y-10 opacity-0 transition-all duration-300`;
    
    if (type === 'success') {
        toast.classList.add('bg-emerald-950/80', 'border-emerald-500/40', 'text-emerald-400');
        toast.innerHTML = `<span>✅</span> ${message}`;
    } else if (type === 'error') {
        toast.classList.add('bg-rose-950/80', 'border-rose-500/40', 'text-rose-400');
        toast.innerHTML = `<span>🚨</span> ${message}`;
    } else if (type === 'warning') {
        toast.classList.add('bg-amber-950/80', 'border-amber-500/40', 'text-amber-400');
        toast.innerHTML = `<span>⚠️</span> ${message}`;
    } else {
        toast.classList.add('bg-cyan-950/80', 'border-cyan-500/40', 'text-cyan-400');
        toast.innerHTML = `<span>ℹ️</span> ${message}`;
    }

    document.body.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.classList.remove('translate-y-10', 'opacity-0');
    }, 50);

    // Fade out and remove
    setTimeout(() => {
        toast.classList.add('opacity-0', 'translate-y-10');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}


function renderUnlockableRecipes() {
    const container = document.getElementById('unlockable-recipes-container');
    if (!container) return;
    container.innerHTML = "";

    if (unlockableRecipes.length === 0) {
        container.innerHTML = `<div class="col-span-2 text-center text-xs text-white/30 italic py-4">No unlockable recipes found.</div>`;
        return;
    }

    unlockableRecipes.forEach(recipe => {
        let coverage = Math.round((recipe.coverage || 0) * 100);
        let missing = recipe.missing_count || 0;
        
        const card = document.createElement('div');
        card.className = "mini-card rounded-xl p-3 flex gap-2.5 hover:border-cyan-500/30 cursor-pointer transition relative";
        card.onclick = () => openRecipeModal(recipe, 'unlockable');
        
        card.innerHTML = `
            <!-- Coverage Percent -->
            <div class="flex flex-col items-center justify-center shrink-0 z-10">
                <span class="text-xs font-mono font-bold text-cyan-400 bg-cyan-500/10 px-1.5 py-0.5 rounded">${coverage}%</span>
                <span class="text-[9px] text-red-400 mt-1 uppercase font-bold text-center leading-tight">Missing<br>${missing}</span>
            </div>
            
            <div class="flex-1 min-w-0 z-10">
                <h4 class="text-xs font-semibold text-white truncate leading-tight">${recipe.title}</h4>
                <div class="flex gap-2 text-[9px] text-white/40 mt-1">
                    <span>${recipe.calories || 0} kcal</span>
                    <span>•</span>
                    <span>${recipe.prep_time_min || 20}m</span>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}
