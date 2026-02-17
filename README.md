# QuickReport

QuickReport is a lightweight business analytics web tool for non-technical users.
Upload a CSV or XLSX file and instantly get:
- cleaned dataset preview
- summary statistics
- automatic charts
- plain-English business insight

## Target users
- shop owners
- operations managers
- junior analysts
- anyone who needs quick data understanding without coding

## How to run
1. Make sure Python 3 is installed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

4. Open your browser at:

```text
http://127.0.0.1:5000
```

## Example usage
1. Open QuickReport in your browser.
2. Upload a `.csv` or `.xlsx` sales or inventory file.
3. Review the generated sections:
- Preview (first 10 rows)
- Statistics table
- Histogram and bar chart
- Insight paragraph
