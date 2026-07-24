# AI-Assisted Concrete Mix Design Platform

## Files required
- `app.py`
- `model.pkl`
- `scaler.pkl`
- `requirements.txt`

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
1. Upload all four files to the same GitHub repository.
2. Create a new Streamlit app.
3. Select the repository, branch, and `app.py`.
4. Deploy.

## Important validation CSV rule
The Laboratory Validation tab expects **raw engineering-unit inputs**, not the
previously standardized `test_processed.csv`.

Required feature columns:
- cement
- blast_furnace_slag
- fly_ash
- water
- superplasticizer
- coarse_aggregate
- fine_aggregate
- age

Optional actual-result column:
- concrete_compressive_strength

The app applies `scaler.pkl` internally exactly once.

## Cost and GWP
The included factors are placeholders for demonstration and relative comparison.
Replace them with verified supplier prices and EPD values before practical use.
