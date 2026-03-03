# Document RAG (Retrieval-Augmented Generation)
Implementasi dari Retrieval-Augmented Generation (RAG) untuk document

## Prasyarat Sistem (Prerequisites)
Sebelum menjalankan program, pastikan perangkat sudah memiliki:
1.  **NVIDIA Driver:** Versi terbaru yang mendukung CUDA. [Panduan Instalasi](https://www.nvidia.com/en-us/drivers/).
2.  **Docker & Docker Compose:** Untuk kontainerisasi layanan. [Dokumentasi Docker](https://docs.docker.com/get-started/get-docker/).
3.  **NVIDIA Container Toolkit:** Agar Docker dapat mengakses GPU. [Instruksi Instalasi](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).


## Spesifikasi Model
Sistem menggunakan model LLM yang telah dioptimasi untuk VRAM terbatas : 
* **LLM:** [Llama-3.2-3B-Instruct-AWQ](https://huggingface.co/AMead10/Llama-3.2-3B-Instruct-AWQ) via Hugging Face.


## Panduan Instalasi Model
Sistem menggunakan vLLM untuk inference, model akan dijalakan pada container terpisah untuk memaksimalkan penggunaan GPU.
### 1. Persiapan Model
Gunakan perintah berikut untuk mendapatkan image (pull) vLLM dan menjalankan model Llama-3.1-8B-Instruct-AWQ-INT4. Perintah ini akan otomatis mengunduh model dari Hugging Face pada dijalankan pertama kali.
```bash
    docker run --runtime nvidia --gpus all \
        -v hf_cache:/root/.cache/huggingface \
        -p 8000:8000 \
        --ipc=host \
        vllm/vllm-openai:latest \
        --model AMead10/Llama-3.2-3B-Instruct-AWQ \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.8
```


### Referensi
1. [25 chunking tricks for RAG that devs actually use](https://medium.com/@dev_tips/25-chunking-tricks-for-rag-that-devs-actually-use-12bebd0375bc)
2. [Open Source Embedding Models](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)