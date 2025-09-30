import streamlit as st
import joblib, json, os, inspect, torch, glob
import pandas as pd
from models.model_jovenes import JovenesMLP
from torchinfo import summary as tsummary
from utils.visualizations import plot_confusion_matrix
import pandas as pd

# --- 1) Carga configuración y paths ---
cfg   = json.load(open("configs/config_jovenes.json"))
paths = json.load(open("configs/paths.json"))

# --- 2) Reconstruye mapping_dict igual que en ETL ---
mapping_dict = {}
for xlsx in glob.glob("data/metadata/*.xlsx"):
    sheets = pd.read_excel(xlsx, sheet_name=None)
    for var, df_map in sheets.items():
        if {"Código","Etiqueta"}.issubset(df_map.columns):
            mapping_dict[var] = df_map.set_index("Código")["Etiqueta"].to_dict()

# --- 3) Carga preprocesador y label encoder ---
preproc = joblib.load(os.path.join(paths["processed_data_dir"], "preprocessor.pkl"))
le      = joblib.load(os.path.join(paths["processed_data_dir"], "label_encoder.pkl"))

# --- 4) Reconstruye el modelo ---
cat_cols = cfg["etl"]["cat_cols"]
num_cols = cfg["etl"]["num_cols"]

# 4.1) Calcula in_dim a partir de get_feature_names_out()
feature_names = preproc.get_feature_names_out()
in_dim = len(feature_names)

mcfg = cfg["model"]
model = JovenesMLP(
    input_dim   = in_dim,
    hidden_dims = mcfg["hidden_dims"],
    attn_dim    = mcfg["attn_dim"],
    num_classes = len(le.classes_),
    dropout_p   = mcfg.get("dropout_p", 0.0)
)

# --- 4.2) Carga de pesos robusta ---
model_dir = cfg["outputs"]["model_dir"]
candidates = ["best_model.pt", "jovenes_best.pt"]
ckpt_path = None
for fname in candidates:
    p = os.path.join(model_dir, fname)
    if os.path.isfile(p):
        ckpt_path = p
        break

if ckpt_path is None:
    pts = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(f"No hay checkpoints .pt en {model_dir}")
    pts = sorted(pts, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    ckpt_path = os.path.join(model_dir, pts[-1])

model.load_state_dict(torch.load(ckpt_path))
model.eval()

# --- 5) Sidebar de navegación ---
page = st.sidebar.radio("Página", ["Código", "Arquitectura", "Matriz Confusión", "Inferencia"])

# --- 6) Páginas ---
if page == "Código":
    st.header("Código de la arquitectura")
    src = inspect.getsource(JovenesMLP)
    st.code(src, language="python")

elif page == "Arquitectura":
    st.header("Resumen del modelo")
    try:
        info = tsummary(model, input_size=(1, in_dim), verbose=0).__str__()
    except:
        info = str(model)
    st.text(info)

elif page == "Matriz Confusión":
    st.header("Matriz de Confusión (Test)")
    cm_path = os.path.join(cfg["outputs"]["plots_dir"], "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_column_width=True)
    else:
        st.write("No se encontró la matriz, ejecuta primero el entrenamiento.")

elif page == "Inferencia":
    st.header("Inferir hábito lector")
    # --- formulario de inputs ---
    st.subheader("Variables categóricas")
    input_data = {}
    for var in cat_cols:
        choices = list(mapping_dict.get(var, {}).values())
        choice = st.selectbox(f"{var}", choices)
        input_data[f"{var}_txt"] = choice

    st.subheader("Variables numéricas")
    for var in num_cols:
        input_data[var] = st.number_input(f"{var}", value=0.0)

    if st.button("Clasificar"):
        df = pd.DataFrame([input_data])
        cols_for_transform = [f"{v}_txt" for v in cat_cols] + num_cols
        X  = preproc.transform(df[cols_for_transform])
        t  = torch.from_numpy(X.astype("float32"))
        with torch.no_grad():
            logits = model(t)
            probs  = torch.softmax(logits, dim=1).numpy().flatten()
            pred   = int(probs.argmax())

        st.markdown(f"**Predicción:** `{le.inverse_transform([pred])[0]}`")
        dfp = pd.DataFrame({
            "clase": le.classes_,
            "prob":  probs
        })
        st.bar_chart(dfp.set_index("clase"))
