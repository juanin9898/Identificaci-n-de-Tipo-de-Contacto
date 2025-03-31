let modeloEntrenado = null; // Variable global para almacenar el modelo

document.addEventListener('DOMContentLoaded', function () {
    // Inicializar el dataset si no existe en localStorage
    if (!localStorage.getItem('dataset')) {
        const datasetInicial = [
            { descripcion: "El trabajador sufrió una lesión lumbar tras levantar una caja pesada.", tipo: "Sobretensión/Sobre-esfuerzo" },
            { descripcion: "El trabajador quedó atrapado entre dos máquinas.", tipo: "Atrapado entre o debajo" },
            { descripcion: "Un objeto en movimiento golpeó a la persona.", tipo: "Golpeado por" }
        ];
        localStorage.setItem('dataset', JSON.stringify(datasetInicial));
        console.log("Dataset inicial cargado en localStorage.");
    }

    // Agregar evento para procesar con Ctrl+Enter en el textarea
    document.getElementById("descripcion").addEventListener("keydown", function(event) {
        if (event.key === "Enter" && event.ctrlKey) {
            event.preventDefault();
            identificarContacto();
        }
    });
});

function preprocesarDescripcion(descripcion) {
    // Convertir a minúsculas
    descripcion = descripcion.toLowerCase();

    // Eliminar palabras comunes (stopwords)
    const stopwords = ["el", "la", "los", "las", "de", "del", "en", "por", "para", "con", "que", "y", "a", "un", "una", "al", "se", "su", "sus"];
    descripcion = descripcion.split(/\W+/).filter(word => !stopwords.includes(word)).join(' ');

    // Retornar la descripción procesada
    return descripcion;
}

async function entrenarModelo() {
    const dataset = JSON.parse(localStorage.getItem('dataset')) || [];

    if (dataset.length === 0) {
        alert("El dataset está vacío. Agrega datos antes de entrenar el modelo.");
        throw new Error("El dataset está vacío.");
    }

    const descripciones = dataset.map(item => item.descripcion);
    const etiquetas = dataset.map(item => item.tipo);

    const vocabulario = new Set(descripciones.join(' ').toLowerCase().split(/\W+/));
    const vocabArray = Array.from(vocabulario);

    const descripcionesTensor = descripciones.map(desc => {
        const vector = new Array(vocabArray.length).fill(0);
        preprocesarDescripcion(desc).split(/\W+/).forEach(word => {
            const index = vocabArray.indexOf(word);
            if (index !== -1) vector[index] = 1;
        });
        return vector;
    });

    const tiposUnicos = Array.from(new Set(etiquetas));
    if (tiposUnicos.length === 0) {
        alert("No se encontraron tipos únicos en el dataset.");
        throw new Error("No se encontraron tipos únicos en el dataset.");
    }

    const etiquetasTensor = etiquetas.map(tipo => tiposUnicos.indexOf(tipo));

    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [vocabArray.length], units: 128, activation: 'relu' })); // Más neuronas
    model.add(tf.layers.dense({ units: 64, activation: 'relu' })); // Capa adicional
    model.add(tf.layers.dense({ units: tiposUnicos.length, activation: 'softmax' }));

    model.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });

    const xs = tf.tensor2d(descripcionesTensor);
    const ys = tf.tensor1d(etiquetasTensor, 'int32').toFloat();

    await model.fit(xs, ys, {
        epochs: 50, // Aumentar el número de épocas
        batchSize: 8
    });

    console.log('Modelo entrenado');
    return { model, vocabArray, tiposUnicos };
}

function mostrarModal() {
    const modal = document.getElementById('resultado-modal');
    modal.style.display = 'flex'; // Mostrar el modal
}

function cerrarModal() {
    const modal = document.getElementById('resultado-modal');
    modal.style.display = 'none'; // Ocultar el modal
}

async function identificarContacto() {
    const descripcion = document.getElementById('descripcion').value.trim();
    const resultado = document.getElementById('resultado');

    if (!descripcion) {
        alert("Por favor, ingresa una descripción del caso.");
        return;
    }

    resultado.value = "Procesando...";

    try {
        if (!modeloEntrenado) {
            modeloEntrenado = await entrenarModelo();
        }

        const { model, vocabArray, tiposUnicos } = modeloEntrenado;
        const vector = new Array(vocabArray.length).fill(0);
        preprocesarDescripcion(descripcion).split(/\W+/).forEach(word => {
            const index = vocabArray.indexOf(word);
            if (index !== -1) vector[index] = 1;
        });

        const inputTensor = tf.tensor2d([vector], undefined, 'float32');
        const prediction = model.predict(inputTensor);
        const tipoIndex = prediction.argMax(-1).dataSync()[0];

        resultado.value = `Tipo identificado: ${tiposUnicos[tipoIndex]}`;
    } catch (error) {
        console.error("Error al procesar la descripción:", error);
        resultado.value = "Hubo un error al procesar la información. Inténtalo nuevamente.";
    }
}

async function marcarCorrecto() {
    const descripcion = document.getElementById('descripcion').value.trim();
    const resultado = document.getElementById('resultado').value.trim();

    if (!descripcion || !resultado) {
        alert("Por favor, ingresa una descripción y verifica el resultado antes de confirmar.");
        return;
    }

    try {
        let dataset = JSON.parse(localStorage.getItem('dataset')) || [];
        const existe = dataset.some(item => item.descripcion === descripcion && item.tipo === resultado);
        if (existe) {
            alert("Este caso ya está registrado como correcto.");
            return;
        }

        dataset.push({ descripcion, tipo: resultado });
        localStorage.setItem('dataset', JSON.stringify(dataset));
        console.log("Caso correcto añadido al dataset:", { descripcion, tipo: resultado });

        alert("El caso ha sido registrado como correcto.");
    } catch (error) {
        console.error("Error al registrar el caso correcto:", error);
        alert("Hubo un error al registrar el caso. Inténtalo nuevamente.");
    }
}

async function marcarIncorrecto() {
    const descripcion = document.getElementById('descripcion').value.trim();
    const correccion = document.getElementById('correccion').value;

    if (!descripcion || !correccion) {
        alert("Por favor, ingresa una descripción y selecciona un tipo correcto.");
        return;
    }

    try {
        let dataset = JSON.parse(localStorage.getItem('dataset')) || [];
        dataset.push({ descripcion, tipo: correccion });
        localStorage.setItem('dataset', JSON.stringify(dataset));
        console.log("Nueva entrada añadida al dataset:", { descripcion, tipo: correccion });

        modeloEntrenado = await entrenarModelo();
        alert("El modelo ha sido reentrenado con la corrección.");
    } catch (error) {
        console.error("Error al procesar la corrección:", error);
        alert("Hubo un error al procesar la corrección. Inténtalo nuevamente.");
    }
}

async function cargarCasosEnLote() {
    try {
        const response = await fetch('bulk_dataset.json');
        const nuevosCasos = await response.json();

        let dataset = JSON.parse(localStorage.getItem('dataset')) || [];
        nuevosCasos.forEach(caso => {
            const existe = dataset.some(item => item.descripcion === caso.descripcion && item.tipo === caso.tipo);
            if (!existe) {
                dataset.push(caso);
            }
        });

        localStorage.setItem('dataset', JSON.stringify(dataset));
        console.log("Casos en lote añadidos al dataset:", nuevosCasos);

        modeloEntrenado = await entrenarModelo();
        alert("El modelo ha sido reentrenado con los nuevos casos.");
    } catch (error) {
        console.error("Error al cargar los casos en lote:", error);
        alert("Hubo un error al cargar los casos en lote. Inténtalo nuevamente.");
    }
}