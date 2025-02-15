from transformers import (
    CamembertTokenizerFast,
    CamembertModel,
    AutoModel,
    AutoTokenizer,
    BertTokenizerFast,
    BertModel,
    AlbertTokenizerFast,
    AlbertModel,
)

TYPEMODELS = [
    "biobert_large",
    "clinicalbert",
    "pubmedbert",
    "camembert_large",
    "drbert_large",
    "camembert_bio",
    "multilingual_bert",
    "fralbert",
    "sapbert",
    "stella",
    "kalm",
    "robertabi",
    "jina",
    "stellabig",
    "e5",
    "modernbert",
    "qwen2",
    "gte",
]


def get_models(typem=None):
    models, tokenizers, types_models = [], [], []

    if typem is None:
        typemodel = TYPEMODELS
    elif type(typem) == str:
        typemodel = [typem]
    elif type(typem) == list:
        typemodel = typem
    else:
        print("Inappropriate typemodel ! Exit now.")
        exit()

    for t in typemodel:
        if t not in TYPEMODELS:
            print(f"One inappropriate typemodel: {t} ! Exit now.")
            exit()

    if "biobert_large" in typemodel:
        biob_tokenizer = AutoTokenizer.from_pretrained(
            "dmis-lab/biobert-large-cased-v1.1"
        )
        biob_model = AutoModel.from_pretrained("dmis-lab/biobert-large-cased-v1.1")
        models.append(biob_model)
        biob_tokenizer.model_max_length = biob_model.config.max_position_embeddings
        tokenizers.append(biob_tokenizer)
        types_models.append("biobert_large")

    if "clinicalbert" in typemodel:
        clin_tokenizer = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        clin_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        models.append(clin_model)
        clin_tokenizer.model_max_length = clin_model.config.max_position_embeddings - 2
        tokenizers.append(clin_tokenizer)
        types_models.append("clinicalbert")

    if "pubmedbert" in typemodel:
        pubmed_tokenizer = AutoTokenizer.from_pretrained(
            "neuml/pubmedbert-base-embeddings"
        )
        pubmed_model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
        models.append(pubmed_model)
        pubmed_tokenizer.model_max_length = (
            pubmed_model.config.max_position_embeddings - 2
        )
        tokenizers.append(pubmed_tokenizer)
        types_models.append("pubmedbert")

    if "camembert_large" in typemodel:
        cam_tokenizer = CamembertTokenizerFast.from_pretrained(
            "almanach/camembert-large"
        )
        cam_model = CamembertModel.from_pretrained("almanach/camembert-large")
        models.append(cam_model)
        cam_tokenizer.model_max_length = cam_model.config.max_position_embeddings - 2
        tokenizers.append(cam_tokenizer)
        types_models.append("camembert_large")

    if "drbert_large" in typemodel:
        drbert_large_tokenizer = AutoTokenizer.from_pretrained(
            "Dr-BERT/DrBERT-7GB-Large"
        )
        drbert_large_model = AutoModel.from_pretrained("Dr-BERT/DrBERT-7GB-Large")
        models.append(drbert_large_model)
        drbert_large_tokenizer.model_max_length = (
            drbert_large_model.config.max_position_embeddings - 2
        )
        tokenizers.append(drbert_large_tokenizer)
        types_models.append("drbert_large")

    if "camembert_bio" in typemodel:
        cambio_tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-bio-base")
        cambio_model = AutoModel.from_pretrained("almanach/camembert-bio-base")
        models.append(cambio_model)
        cambio_tokenizer.model_max_length = (
            cambio_model.config.max_position_embeddings - 2
        )
        tokenizers.append(cambio_tokenizer)
        types_models.append("camembert_bio")

    if "multilingual_bert" in typemodel:
        mbert_tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-multilingual-cased"
        )
        mbert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
        models.append(mbert_model)
        tokenizers.append(mbert_tokenizer)
        types_models.append("multilingual_bert")

    if "fralbert" in typemodel:
        albert_tokenizer = AlbertTokenizerFast.from_pretrained(
            "cservan/fralbert-base-cased"
        )
        albert_model = AlbertModel.from_pretrained("cservan/fralbert-base-cased")
        models.append(albert_model)
        tokenizers.append(albert_tokenizer)
        types_models.append("fralbert")

    if "sapbert" in typemodel:
        sapbert_tokenizer = AutoTokenizer.from_pretrained(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )
        sapbert_model = AutoModel.from_pretrained(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )
        models.append(sapbert_model)
        sapbert_tokenizer.model_max_length = (
            sapbert_model.config.max_position_embeddings
        )
        tokenizers.append(sapbert_tokenizer)
        types_models.append("sapbert")

    if "stella" in typemodel:
        stella_model = AutoModel.from_pretrained(
            "dunzhang/stella_en_400M_v5", trust_remote_code=True
        )
        stella_tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_400M_v5")
        models.append(stella_model)
        stella_tokenizer.model_max_length = stella_model.config.max_position_embeddings
        tokenizers.append(stella_tokenizer)
        types_models.append("stella")

    if "kalm" in typemodel:
        kalm_model = AutoModel.from_pretrained(
            "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1"
        )
        kalm_tokenizer = AutoTokenizer.from_pretrained(
            "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v1"
        )
        models.append(kalm_model)
        kalm_tokenizer.model_max_length = kalm_model.config.max_position_embeddings
        tokenizers.append(kalm_tokenizer)
        types_models.append("kalm")

    if "robertabi" in typemodel:
        robertabi_model = AutoModel.from_pretrained(
            "Lajavaness/bilingual-embedding-large", trust_remote_code=True
        )
        robertabi_tokenizer = AutoTokenizer.from_pretrained(
            "Lajavaness/bilingual-embedding-large"
        )
        models.append(robertabi_model)
        robertabi_tokenizer.model_max_length = (
            robertabi_model.config.max_position_embeddings - 2
        )
        tokenizers.append(robertabi_tokenizer)
        types_models.append("robertabi")

    if "jina" in typemodel:
        jina_model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        )
        jina_tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
        models.append(jina_model)
        jina_tokenizer.model_max_length = jina_model.config.max_position_embeddings
        tokenizers.append(jina_tokenizer)
        types_models.append("jina")

    if "gte" in typemodel:
        gte_tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5")
        gte_model = AutoModel.from_pretrained(
            "Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
        )
        models.append(gte_model)
        gte_tokenizer.model_max_length = gte_model.config.max_position_embeddings - 2
        tokenizers.append(gte_tokenizer)
        types_models.append("gte")

    if "stellabig" in typemodel:
        stellabig_model = AutoModel.from_pretrained(
            "dunzhang/stella_en_1.5B_v5", trust_remote_code=True
        )
        stellabig_tokenizer = AutoTokenizer.from_pretrained(
            "dunzhang/stella_en_1.5B_v5", trust_remote_code=True
        )
        stellabig_tokenizer.model_max_length = (
            stellabig_model.config.max_position_embeddings
        )
        models.append(stellabig_model)
        tokenizers.append(stellabig_tokenizer)
        types_models.append("stellabig")

    if "e5" in typemodel:
        e5_model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")
        e5_tokenizer = AutoTokenizer.from_pretrained(
            "intfloat/multilingual-e5-large-instruct"
        )
        models.append(e5_model)
        tokenizers.append(e5_tokenizer)
        types_models.append("e5")

    if "modernbert" in typemodel:
        modbert_model = AutoModel.from_pretrained("nomic-ai/modernbert-embed-base")
        modbert_tokenizer = AutoTokenizer.from_pretrained(
            "nomic-ai/modernbert-embed-base"
        )
        models.append(modbert_model)
        tokenizers.append(modbert_tokenizer)
        types_models.append("modernbert")

    if "qwen2" in typemodel:
        qwen_model = AutoModel.from_pretrained(
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True
        )
        qwen_tokenizer = AutoTokenizer.from_pretrained(
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True
        )
        models.append(qwen_model)
        tokenizers.append(qwen_tokenizer)
        types_models.append("qwen2")
    return models, tokenizers, types_models
