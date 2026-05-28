document.addEventListener("DOMContentLoaded", () => {

    // 1. Read Selected Mode from LocalStorage
    const modelMode = localStorage.getItem("selectedModel") || "COMBINED_512"; 
    
    const titleMap = {
        'COMBINED_512': 'Standard Pipeline (512px)',
        'SEG_300': 'Segmentation Pipeline (300px)'
    };
    
    const modelTextElement = document.getElementById("selectedModelText");
    if (modelTextElement) {
        modelTextElement.textContent = titleMap[modelMode] || modelMode;
    }

    // Hide Grad-CAM checkbox if using Segmentation mode
    if (modelMode === 'SEG_300') {
        const gc = document.getElementById("gradcamContainer");
        if(gc) gc.style.display = 'none';
    }

        // Activate grid layout for segmentation mode
    if (modelMode === 'SEG_300') {
        const container = document.querySelector(".comparison-container");
        if (container) {
            container.classList.add("seg-mode");
        }
    }

    // --- DOM REFERENCES ---
    const resultsContainer = document.getElementById("resultsContainer");
    const errorMsgDiv = document.getElementById("res-error-msg");
    const latText = document.getElementById("res-laterality");
    // "fileName" is used for Status Updates in the results section
    const statusText = document.getElementById("fileName") || document.getElementById("file-name"); 

    // Grids
    const grid512 = document.getElementById("grid-512");
    const grid300 = document.getElementById("grid-300");

    // Images containers
    const boxOriginal = document.getElementById("boxOriginal");
    const boxGradcam = document.getElementById("boxGradcam");
    const boxSegmented = document.getElementById("boxSegmented");
    const boxCropped = document.getElementById("boxCropped");
    
    // --- NUEVO: Referencia al contenedor YOLO ---
    const boxYolo = document.getElementById("boxYolo"); 
    
    // Image elements
    const imgOriginal = document.getElementById("visual_original");
    const imgGradcam = document.getElementById("imgGradcam"); 
    const imgSegmented = document.getElementById("visual_segmented");
    const imgCropped = document.getElementById("visual_cropped");
    
    // --- NUEVO: Referencia a la imagen YOLO ---
    const imgYolo = document.getElementById("visual_yolo");

    // --- UPLOAD ELEMENTS (Matches new HTML) ---
    const realInput = document.getElementById("file-input");
    const dropArea = document.getElementById("drop-area");
    const browseBtn = document.querySelector(".browse-btn");
    const fileNameDisplay = document.getElementById("file-name");

    // --- EVENT LISTENERS ---

    // 1. Browse Button
    if (browseBtn) {
        browseBtn.addEventListener("click", (e) => { 
            e.stopPropagation(); 
            realInput.click(); 
        });
    }

    // 2. Drop Area Interactions
    if (dropArea) {
        dropArea.addEventListener("click", () => { realInput.click(); });
        
        dropArea.addEventListener("dragover", (e) => { 
            e.preventDefault(); 
            dropArea.classList.add("dragover"); 
        });
        
        dropArea.addEventListener("dragleave", () => { 
            dropArea.classList.remove("dragover"); 
        });
        
        dropArea.addEventListener("drop", (e) => {
            e.preventDefault(); 
            dropArea.classList.remove("dragover");
            if (e.dataTransfer.files[0]) processUpload(e.dataTransfer.files[0]);
        });
    }

    realInput.addEventListener("change", () => { 
        if (realInput.files.length > 0) 
            processUpload(realInput.files[0]); 
    });

    function setupCollapsibles() {
    const boxes = document.querySelectorAll(".explanation-box.collapsible");

    boxes.forEach(box => {
        const header = box.querySelector(".explanation-header");
        if (header) {
            header.addEventListener("click", () => {
                box.classList.toggle("active");
            });
        }
    });
}

setupCollapsibles();

    // --- MAIN PROCESSING LOGIC ---
    async function processUpload(file) {
        if (!file) return;

        // --- RESET UI ---
        if (statusText) statusText.textContent = `Processing with ${modelMode}...`;

        if (resultsContainer) resultsContainer.style.display = "none";
        if (errorMsgDiv) {
            errorMsgDiv.style.display = "none";
            errorMsgDiv.textContent = "";
        }

        // Hide Grids
        if (grid512) grid512.style.display = "none";
        if (grid300) grid300.style.display = "none";

        // Hide Boxes
        if (boxOriginal) boxOriginal.style.display = "none";
        if (boxGradcam) boxGradcam.style.display = "none";
        if (boxSegmented) boxSegmented.style.display = "none";
        if (boxCropped) boxCropped.style.display = "none";
        if (boxYolo) boxYolo.style.display = "none";
        const boxCalciumReset = document.getElementById("boxCalcium");
        if (boxCalciumReset) boxCalciumReset.style.display = "none";

        // Clear Images
        if (imgOriginal) imgOriginal.src = "";
        if (imgGradcam) imgGradcam.src = "";
        if (imgSegmented) imgSegmented.src = "";
        if (imgCropped) imgCropped.src = "";
        if (imgYolo) imgYolo.src = "";
        const visualCalciumReset = document.getElementById("visual_calcium");
        if (visualCalciumReset) visualCalciumReset.src = "";
        const calciumBadgeReset = document.getElementById("calciumBadge");
        if (calciumBadgeReset) calciumBadgeReset.textContent = "";

        // Update upload box text
        if (fileNameDisplay) {
            fileNameDisplay.textContent = `Selected: ${file.name}`;
            fileNameDisplay.classList.add('visible');
        }

        const thresholdInput = document.getElementById("thresholdSelect");
        const threshold = thresholdInput ? parseFloat(thresholdInput.value) : 0.5;
        const gradcamInput = document.getElementById("gradcamCheck");
        const gradcamCheck = gradcamInput ? gradcamInput.checked : false;

        // --- RESET UI ---
        if(statusText) statusText.textContent = `Processing with ${modelMode}...`;

        try {
            // ===============================================
            // MODE 1: COMBINED_512 (Standard)
            // ===============================================
            if (modelMode === 'COMBINED_512') {
                console.log("Initializing phase 1: Fast Prediction...");
                if(statusText) statusText.textContent = `Analyzing Patterns...`;
                
                // 1. Fast Phase (Prediction only)
                const formDataFast = new FormData();
                formDataFast.append("file", file);
                formDataFast.append("threshold", threshold);
                formDataFast.append("model_name", modelMode);
                
                // Get manual inputs if they exist (for hybrid model)
                const ageInput = document.getElementById("ageInput");
                const sexInput = document.getElementById("sexInput");
                if(ageInput) formDataFast.append("age", ageInput.value);
                if(sexInput) formDataFast.append("sex", sexInput.value);
                
                const resFast = await fetch("/predict_512_fast", { method: "POST", body: formDataFast });
                const dataFast = await resFast.json();
                
                if (!resFast.ok || dataFast.error) throw new Error(dataFast.error || "Error in Phase 1 Prediction");
                
                // --- SHOW RESULTS PHASE 1 ---
                if(resultsContainer) resultsContainer.style.display = "block";
                if(latText) latText.textContent = dataFast.laterality;
                
                // Show Original
                if (dataFast.image_original && boxOriginal) {
                    boxOriginal.style.display = "flex";
                    imgOriginal.src = dataFast.image_original;
                }

                // Show Grid and Texts
                if(grid512) grid512.style.display = "flex";
                
                const cnnBox = document.getElementById("res-cnn-box");
                if(cnnBox) {
                    document.getElementById("res-cnn-diag").textContent = dataFast.result_cnn.diagnosis;
                    document.getElementById("res-cnn-prob").textContent = dataFast.result_cnn.prob.toFixed(4);
                    updateStatusClass(cnnBox, dataFast.result_cnn.diagnosis);
                }

                const hybBox = document.getElementById("res-hyb-box");
                if(hybBox) {
                    document.getElementById("res-hyb-diag").textContent = dataFast.result_hybrid.diagnosis;
                    document.getElementById("res-hyb-prob").textContent = dataFast.result_hybrid.prob.toFixed(4);
                    updateStatusClass(hybBox, dataFast.result_hybrid.diagnosis);
                }

                // Ejecutar YOLO solo si el clasificador dice positivo
                if (dataFast.result_cnn.diagnosis === "Tendinopathy Detected") {
                    runYolo(file);
                }

                // 2. Slow Phase (Grad-CAM in background)
                if (gradcamCheck) {
                    console.log("Phase 2: Heatmap...");
                    if(statusText) statusText.textContent = `Diagnosis Ready. Heatmap processing...`;

                    const loader = document.getElementById("gradcamLoader");
                    const img = document.getElementById("imgGradcam");

                    if (boxGradcam) {
                        boxGradcam.style.display = "none";
                    }

                    if (loader) {
                        loader.style.display = "flex"; 
                        if(img) img.style.display = "none";
                    }

                    const formDataGrad = new FormData();
                    formDataGrad.append("file", file);
                    
                    try {
                        const resGrad = await fetch("/predict_512_gradcam", { method: "POST", body: formDataGrad });
                        const dataGrad = await resGrad.json();
                        
                        if (dataGrad.success && dataGrad.gradcam_preview && img) {

                            img.onload = () => {
                                if (boxGradcam) boxGradcam.style.display = "flex";
                                if (loader) loader.style.display = "none";
                                img.style.display = "block";

                                img.classList.remove("pop-in");
                                void img.offsetWidth;
                                img.classList.add("pop-in");

                                if (statusText) statusText.textContent = `Analyzed: ${file.name}`;
                            };

                            img.src = dataGrad.gradcam_preview;

                        } else {
                            if (boxGradcam) boxGradcam.style.display = "none";
                        }
                    } catch (e) {
                        console.error("Error at fetching GradCAM:", e);
                        if(boxGradcam) boxGradcam.style.display = "none";
                    }
                } else {
                    if(boxGradcam) boxGradcam.style.display = "none";
                }

            } else {
                // ===============================================
                // MODE 2: SEG_300 (Segmentation)
                // ===============================================
                console.log("Initializing 300px Segmentation Mode...");
                
                if(grid300) grid300.style.display = "grid"; 
                const boxGrad300 = document.getElementById("boxGradcam300");
                const imgGrad300 = document.getElementById("imgGradcam300");
                const loader300 = document.getElementById("gradcamLoader300");

                if (boxGrad300) {
                    boxGrad300.style.display = "none";
                }

                if (loader300) loader300.style.display = "flex";
                if (imgGrad300) imgGrad300.style.display = "none";

                // Main Fetch
                const formDataSeg = new FormData();
                formDataSeg.append("file", file);
                formDataSeg.append("threshold", threshold);
                formDataSeg.append("model_name", modelMode);
                            
                const res = await fetch("/predict", { method: "POST", body: formDataSeg });
                const data = await res.json();
                
                if (!res.ok || data.error) throw new Error(data.error || "Error in prediction");
            
                if(statusText) statusText.textContent = `Analyzed: ${file.name}`;
                if(latText) latText.textContent = data.laterality;
                if(resultsContainer) resultsContainer.style.display = "block";

                if (data.image_original && boxOriginal) {
                    boxOriginal.style.display = "flex";
                    imgOriginal.src = data.image_original;
                }
                
                const segBox = document.getElementById("res-seg-box");
                if(segBox) {
                    document.getElementById("res-seg-diag").textContent = data.result_seg.diagnosis;
                    document.getElementById("res-seg-prob").textContent = data.result_seg.prob.toFixed(4);
                    updateStatusClass(segBox, data.result_seg.diagnosis);
                }

                if (data.image_segmented && boxSegmented) {
                    boxSegmented.style.display = "flex";
                    imgSegmented.src = data.image_segmented;
                }
                if (data.image_cropped && boxCropped) {
                    boxCropped.style.display = "flex";
                    imgCropped.src = data.image_cropped;
                }

                // Ejecutar YOLO solo si el clasificador dice positivo
                if (data.result_seg.diagnosis === "Tendinopathy Detected") {
                    runYolo(file);
                }

                // 3. FETCH GRAD-CAM 300 (Background)
                console.log("Phase 2: Heatmap 300px...");
                const formDataGrad300 = new FormData();
                formDataGrad300.append("file", file);

                try {
                    const resGrad = await fetch("/predict_300_gradcam", { method: "POST", body: formDataGrad300 });
                    const dataGrad = await resGrad.json();

                    if (dataGrad.success && dataGrad.gradcam_preview && imgGrad300) {

                        imgGrad300.onload = () => {
                            if (boxGrad300) boxGrad300.style.display = "flex";
                            if (loader300) loader300.style.display = "none";
                            imgGrad300.style.display = "block";

                            imgGrad300.classList.remove("pop-in");
                            void imgGrad300.offsetWidth;
                            imgGrad300.classList.add("pop-in");
                        };

                        imgGrad300.src = dataGrad.gradcam_preview;

                    } else {
                        if (boxGrad300) boxGrad300.style.display = "none";
                    }
                } catch (e) {
                    console.error("Error GradCAM 300:", e);
                    if(boxGrad300) boxGrad300.style.display = "none";
                }
            }

        } catch (error) {
            console.error("General Error:", error);
            if(resultsContainer) resultsContainer.style.display = "block";
            if(errorMsgDiv) {
                errorMsgDiv.textContent = `Error: ${error.message}`;
                errorMsgDiv.style.display = "block";
            }
            if(statusText) statusText.textContent = "Error during analysis";
        }
    }
    
    // --- FUNCIÓN AUXILIAR PARA YOLO + CALCIUM ---
    async function runYolo(file) {

        const yoloText = document.getElementById("yolo-text");
        const calciumText = document.getElementById("calcium-text");
        const debugEl = document.getElementById("res-error-msg"); // Elemento que lee el PDF

        if (!boxYolo || !imgYolo) return;

        // Nueva sección calcium
        const boxCalcium = document.getElementById("boxCalcium");
        const imgCalcium = document.getElementById("img-calcium");

        // 1. Ocultar cajas al inicio
        if (debugEl) debugEl.innerText = ""; // Limpiar debug anterior
        boxYolo.style.display = "none";
        if (boxCalcium) boxCalcium.style.display = "none";

        if (boxCalcium) {
            boxCalcium.style.display = "none";
        }

        const container = document.querySelector(".comparison-container");
        if (container) {
            container.classList.remove("no-yolo");
        }

        const fd = new FormData();
        fd.append("file", file);

        try {

            const res = await fetch("/predict_yolo", {
                method: "POST",
                body: fd
            });

            const data = await res.json();
            console.log("FULL YOLO RESPONSE:", data);

            if (!data.success) return;

            if (debugEl && data.debug_scores) {
                debugEl.innerText = data.debug_scores; 
            }

            const info = data.model_info || "";

            // -------------------------------------------------
            // 1️⃣ FILTRO: SI NO HAY CALCIFICACIÓN
            // -------------------------------------------------
            if (info.includes("No Calcification")) {
                console.log("YOLO: Paciente sano. Caja oculta.");
                return;
            }

            // -------------------------------------------------
            // 2️⃣ MOSTRAR IMAGEN YOLO
            // -------------------------------------------------
            if (data.image_yolo) {

                imgYolo.src = data.image_yolo;

                if (yoloText) {
                    yoloText.textContent = info;
                    yoloText.style.color = "#dc2626";
                    imgYolo.style.border = "2px solid #dc2626";
                }

                boxYolo.style.display = "flex";
            }

            // -------------------------------------------------
            // 3️⃣ MOSTRAR IMAGEN CALCIUM (SI EXISTE)
            // -------------------------------------------------
            console.log("CHECK:", data.image_calcium, data.prediction);
            if (data.image_calcium && data.prediction) {

                const imgCalcium = document.getElementById("visual_calcium");
                const boxCalcium = document.getElementById("boxCalcium");
                const calciumBadge = document.getElementById("calciumBadge");

                if (imgCalcium) {
                    imgCalcium.src = data.image_calcium;
                }

                if (calciumBadge) {
                    calciumBadge.textContent = data.prediction;
                }

                if (boxCalcium) {
                    boxCalcium.style.display = "flex";
                }
            }

        } catch (e) {
            console.error("Error YOLO:", e);
            boxYolo.style.display = "none";
            if (boxCalcium) boxCalcium.style.display = "none";
        }
    }

    function updateStatusClass(element, diagnosisText) {
        if (!element) return;
        element.classList.remove("status-detected", "status-normal");
        
        if (diagnosisText === "Normal" || diagnosisText === "No Tendinopathy Detected") {
            element.classList.add("status-normal");
        } else {
            element.classList.add("status-detected");
        }
    }

    // Modal Logic
    function setupImageModal() {
        const modal = document.getElementById("imageModal");
        const modalImg = document.getElementById("imgExpanded");
        const captionText = document.getElementById("caption");
        const closeBtn = document.querySelector(".close-modal");

        if (!modal) return;

        const imagesToEnlarge = [
            document.getElementById("visual_original"),
            document.getElementById("imgGradcam"),
            document.getElementById("visual_segmented"),
            document.getElementById("visual_cropped"),
            document.getElementById("imgGradcam300"),
            document.getElementById("visual_yolo"), 
            document.getElementById("visual_calcium")
        ];

        imagesToEnlarge.forEach(img => {
            if (img) {
                img.addEventListener("click", function() {
                    if (this.src && this.src !== "" && this.style.display !== "none") {
                        modal.style.display = "block";
                        modalImg.src = this.src;
                        
                        const label = this.parentElement.querySelector(".img-label") || this.parentElement.querySelector(".img-title");
                        captionText.innerHTML = label ? label.textContent : "Expanded View";
                    }
                });
            }
        });

        if (closeBtn) {
            closeBtn.onclick = function() { 
                modal.style.display = "none"; 
            }
        }

        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = "none";
            }
        }
        
        document.addEventListener('keydown', function(event) {
            if (event.key === "Escape" && modal.style.display === "block") {
                modal.style.display = "none";
            }
        });
    }
    setupImageModal();
});