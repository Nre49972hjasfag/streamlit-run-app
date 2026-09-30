import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE SETUP ---
st.set_page_config(page_title="Deep Quant Dashboard", layout="wide", page_icon="📈")
st.title("📈 Recurrent Deep Learning Quant Predictor")
st.markdown("Train and evaluate a real-time **Gated Recurrent Unit (GRU)** PyTorch neural network on historical assets.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🕹️ Control Settings")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2026-01-01"))

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model Hyperparameters")
lookback = st.sidebar.slider("Lookback Window (Days)", min_value=3, max_value=30, value=10)
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=100, value=20)
batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")

# --- MODEL DEFINITION ---
class GRUMarketPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(GRUMarketPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return self.sigmoid(out)

# --- SEQUENCE CREATION UTILITY ---
def create_sequences(data, targets, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(targets[i + seq_length - 1])
    return np.array(xs), np.array(ys)

# --- PIPELINE TRIGGER ---
if st.sidebar.button("🚀 Run Pipeline"):
    # 1. FETCH DATA
    with st.spinner("Downloading technical data arrays..."):
        df = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
    if len(df) < 50:
        st.error("Insufficient asset history found for the parameters requested.")
    else:
        # 2. FEATURE ENGINEERING
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df['Return'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean() / df['Close'] - 1
        df['MA20'] = df['Close'].rolling(window=20).mean() / df['Close'] - 1
        df['Volatility'] = df['Return'].rolling(window=10).std()
        df['Momentum'] = df['Close'] / df['Close'].shift(5) - 1
        df.dropna(inplace=True)

        feature_cols = ['Return', 'MA5', 'MA20', 'Volatility', 'Momentum']
        X_raw = df[feature_cols].values
        y_raw = df['Target'].values

        # 3. SPLIT & SCALE
        split_idx = int(len(X_raw) * 0.8)
        X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
        y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 4. SEQUENCES & TENSORS
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, lookback)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw, lookback)

        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        # 5. INITIALIZE & TRAIN
        model = GRUMarketPredictor(input_dim=len(feature_cols), hidden_dim=32, num_layers=2)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        loss_history = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Training Network Matrix... Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        status_text.success("Neural Network Fully Trained! Generating Metrics UI...")

        # 6. EVALUATION
        model.eval()
        with torch.no_grad():
            test_probabilities = model(X_test_tensor).numpy()
            test_predictions = (test_probabilities > 0.5).astype(int)

        # --- LAYOUT DASHBOARD VISUALIZATIONS ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Convergence Curve")
            fig_loss = px.line(x=range(1, epochs + 1), y=loss_history, labels={'x':'Epoch', 'y':'BCELoss Value'})
            st.plotly_chart(fig_loss, use_container_width=True)

        with col2:
            st.subheader("🧮 Confusion Matrix")
            cm = confusion_matrix(y_test_seq, test_predictions)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted Signal", y="Actual Direction"),
                               x=['Down Trend', 'Up Trend'], y=['Down Trend', 'Up Trend'], color_continuous_scale="Blues")
            st.plotly_chart(fig_cm, use_container_width=True)

        # Metric Summaries
        report = classification_report(y_test_seq, test_predictions, output_dict=True, target_names=['Down', 'Up'])
        st.subheader("📊 Operational Classification Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Global Accuracy", f"{report['accuracy']:.2%}")
        m_col2.metric("Up Trend Precision", f"{report['Up']['precision']:.2%}")
        m_col3.metric("Down Trend Recall", f"{report['Down']['recall']:.2%}")
else:
    st.info("Adjust the structural constraints in the sidebar panel and click **Run Pipeline** to stream data and compute models.")
