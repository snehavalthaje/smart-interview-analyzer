git clone https://github.com/snehavalthaje/smart-interview-analyzer.git
cd smart-interview-analyzer
pip install -r requirements.txt
python src/preprocessing/feature_extractor.py
python src/training/train_model.py
streamlit run ui/app.py
