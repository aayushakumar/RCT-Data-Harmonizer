# RCT Data Harmonizer

A Streamlit app to harmonize variable names and merge participant-level data from multiple randomized controlled trials (RCTs). Automates fuzzy matching of column names, lets users review and override mappings, and provides summary statistics, visualizations, and exportable harmonized datasets and codebooks.

---

## 🚀 Features

- **Multi‐file Upload**: Upload multiple CSV files from different RCT studies.
- **Fuzzy Variable Mapping**: Auto-detect and align column names to a canonical schema using `rapidfuzz`.
- **Interactive Review**: Override or confirm mappings via a Streamlit interface.
- **Merge & Harmonize**: Rename and combine DataFrames on common keys (e.g. `participant_id`).
- **Sanity-Check Summaries**: Generate descriptive statistics and missing‐value reports.
- **Data Visualizations**: Faceted histograms per variable and source file (using Altair or Matplotlib).
- **One-Click Exports**: Download harmonized CSV and a “codebook lookup” mapping file.

---

## 📋 Repository Structure

```

├── app.py               # Main Streamlit application
├── requirements.txt     # Required Python packages and versions
├── README.md            # This documentation
└── data/                # (Optional) Example or template CSV files

````

---

## ⚙️ Requirements

- Python 3.8+
- Streamlit ≥1.28.0
- pandas ≥2.1.0
- rapidfuzz ≥3.3.0
- matplotlib ≥3.7.2
- altair ≥5.1.0

See `requirements.txt` for pinned versions.

---


## ▶️ Usage

1. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

2. **Upload CSVs**

   * Use the sidebar uploader to select one or more RCT CSV files.

3. **Adjust Settings**

   * Set the fuzzy-match threshold in the sidebar.

4. **Review Mappings**

   * Confirm or override auto-detected canonical mappings.

5. **Harmonize & Merge**

   * Click **🔄 Harmonize and Merge Data** to combine datasets.

6. **Explore Results**

   * Preview the merged DataFrame, view summary statistics, and inspect histograms.

7. **Download Outputs**

   * Use download buttons to export:

     * `harmonized_rct_data.csv`
     * `variable_codebook.csv`

---

## 🛠️ Customization

* **Canonical Variables**
  Edit `get_canonical_variables()` in `app.py` to add or adjust your own variable lists.
* **Merge Key**
  By default, merges on `participant_id`. Modify `harmonize_dataframes()` to change join behavior.
* **Visualization Limits**
  The code plots the first six numeric variables—adjust in `plot_histograms()` as needed.

---
## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Contact

**Aayush Kumar**

* Email: [akuma102@uic.edu](mailto:akuma102@uic.edu)
* GitHub: [github.com/aayushakumar](https://github.com/aayushakumar)
* LinkedIn: [linkedin.com/in/aayushakumars](https://www.linkedin.com/in/aayushakumars)

Feel free to open issues or reach out with questions!
