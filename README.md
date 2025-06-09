![image](https://github.com/user-attachments/assets/18c2c4e3-7823-4107-8cc7-afda5ac0875d)

<p align="center">
  <a href="https://solosolve-ai-demo.lovable.app/" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%9A%80%20Live%20Demo-Online-green?style=for-the-badge" alt="Live Demo"/>
  </a>
  <a href="https://github.com/solosolve-ai/solosolve-ai-demo" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%92%BB%20GitHub%20Repo-Click%20Here-blue?style=for-the-badge" alt="Dev Repo"/>
  </a>
</p>

# **Your SoloSolver**

![demo](https://github.com/solosolve-ai/solosolve-ai/blob/main/docs/solosolve-demo.gif)

An AI-Powered Customer Complaint Resolution System for Amazon Fashion, built on a foundation of SOTA MLOps, advanced data engineering, and state-of-the-art Large Language Models.

---

## 🌟 **Project Vision**

To create a highly accurate, scalable, and continuously improving AI-powered customer complaint resolution system for Amazon Fashion. SoloSolver Pro will leverage multimodal inputs, deep contextual understanding through RAG, robust data engineering practices, and efficient, SOTA LLM fine-tuning and deployment techniques to provide fair, consistent, and empathetic customer support, ultimately enhancing customer satisfaction and operational efficiency.

---

## 🚀 **Core Features**

SoloSolver Pro is engineered with a multi-faceted approach to complaint resolution, moving beyond simple classification to a deep, contextual understanding of each case.

### **1. SOTA LLM-Powered Resolution Engine**
- **Fine-Tuned Gemma-3 Model:** Utilizes a `google/gemma-3-4b-it` model fine-tuned with QLoRA for state-of-the-art performance in generating structured, empathetic, and policy-aligned resolutions.
- **Multi-Task Learning:** The model simultaneously learns to generate a comprehensive JSON analysis, classify complaint drivers, determine severity, and predict sentiment, all from a single input.

### **2. Advanced RAG & Contextual Understanding**
- **Dynamic Policy Retrieval:** Implements a sophisticated Retrieval-Augmented Generation (RAG) pipeline using a semantic vector search (ChromaDB/Vertex AI) on a meticulously chunked Policy Database to ground every decision in company policy.
- **Dynamic User Profiling (UDP):** Generates and queries user-specific profiles on-the-fly, considering their historical interactions, purchase value, and past complaint patterns to provide personalized and fair resolutions.

### **3. Multimodal Complaint Analysis**
- **Conceptual Image Analysis:** Even without direct image processing in the initial phase, the model is prompted to infer the context of user-submitted images based on the complaint text, enabling a more holistic analysis of issues like "item arrived damaged."

### **4. Scalable Data Engineering Backbone**
- **Modern Data Stack:** Leverages Google Cloud Platform (GCS, BigQuery) and `dbt` for a scalable, version-controlled, and testable data pipeline, transforming raw review data into a "gold standard" transaction database.
- **Automated SFT Data Curation:** Employs stratified sampling and SQL UDFs within BigQuery to programmatically curate a high-quality, balanced Supervised Fine-Tuning (SFT) dataset, ensuring model robustness.

---

## 🛠️ **Tech Stack**

Our stack is chosen for scalability, reproducibility, and SOTA performance.

| Category                | Technologies & Tools                                                                         |
|-------------------------|----------------------------------------------------------------------------------------------|
| **Cloud & Data Platform** | Google Cloud Platform (GCP), Google Cloud Storage (GCS), BigQuery                            |
| **Data Engineering & ETL**| `dbt`, Google Dataproc / Spark, Pandas, Parquet                                              |
| **Model & AI Frameworks** | PyTorch, Hugging Face `transformers`, `peft` (QLoRA), `trl` (SFT, DPO), `bitsandbytes` (4-bit Q) |
| **LLM & RAG**             | Google Gemma-3, LangChain/LlamaIndex, ChromaDB, Vertex AI Vector Search                      |
| **MLOps & Deployment**    | Docker, Vertex AI (Experiments, Model Registry, Pipelines), Google Cloud Run (GPU), FastAPI  |
| **Evaluation & Analysis** | `evaluate` (ROUGE, BLEU, BERTScore), Scikit-learn, Matplotlib, Seaborn                        |

---

## 🗂️ **Data Sources**

The core dataset for training and analysis is sourced from:
- **Raw Amazon Fashion Reviews:** `McAuley-Lab/Amazon-Reviews-2023` (config: `raw_review_Amazon_Fashion`)
- **Raw Amazon Fashion Metadata:** `McAuley-Lab/Amazon-Reviews-2023` (config: `raw_meta_Amazon_Fashion`)

---

## 📊 **Evaluation & Metrics**

Model performance is rigorously assessed across both generation and classification tasks to ensure high quality and reliability.

- **Generation Quality:** Measured with **ROUGE-L, BLEU, BERTScore**, and **JSON Schema Adherence Rate**.
- **Classification Accuracy:** Measured with **F1-Score, Precision, Recall, and ROC AUC** for each classification head.
- **Performance:** Includes **inference latency** and **GPU memory usage** benchmarks for the quantized model to ensure deployment feasibility.
- **Qualitative Analysis:** A side-by-side comparison against the base Gemma-3 model to demonstrate the significant improvements from fine-tuning.

---

## 🗺️ **Project Roadmap**

The project is structured in three ambitious phases, from foundational model development to a fully autonomous, learning agent.

| Phase                                     | Duration (Est.)  | Key Objectives & Deliverables                                                                                                                                                            |
|-------------------------------------------|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Phase 1: SOTA Foundation & Core Model** | 3-4 Months       | **Objectives:** Build data foundation, generate SFT dataset, fine-tune & evaluate Gemma-3.<br/>**Deliverables:** Fine-tuned Gemma-3 model (LoRA adapters & heads), comprehensive evaluation report (with radar plots), GCP deployment blueprint, reproducible Docker environment. |
| **Phase 2: Cloud Integration & Advanced RAG** | 3-4 Months       | **Objectives:** Productionize data pipelines, implement advanced RAG, integrate with MLOps tooling.<br/>**Deliverables:** Vertex AI Pipeline for CI/CD/CT, migration to Vertex AI Vector Search, implementation of a reranker, operational dashboards.          |
| **Phase 3: Agentic Capabilities & Continuous Learning** | Ongoing          | **Objectives:** Evolve into an autonomous agent, learn from feedback.<br/>**Deliverables:** LangChain-based agent with ReAct framework, DPO-tuned model from user feedback, a continuous learning loop, exploration of true multimodal models.                 |

---

## 👥 **Collaborators**

- **Shoval Benjer**
- **Adir Amar**
- **Alon Berkovich**

---

## 🌐 **Get Involved**

We value collaboration! Explore our [GitHub Repository](https://github.com/solosolve-ai/solosolve-ai) to contribute, provide feedback, or track our progress. If the repository is private, please contact one of the collaborators for access.

---

Join us on this transformative journey to redefine customer service with SoloSolver Pro. Together, we innovate for better service.
