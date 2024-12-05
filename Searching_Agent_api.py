import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Login to Hugging Face (run this in terminal or as a separate step)
# !huggingface-cli login

# Load the Llama-3.1-8B model
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

def query_llama(prompt):
    result = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)
    return result[0]['generated_text']

def get_category_and_websites(dataset_description):
    prompt = f"Given this dataset description: '{dataset_description}', suggest a category and 3 relevant websites to search for similar data."
    suggestion = query_llama(prompt)
    category, websites = parse_suggestion(suggestion)
    return category, websites

def parse_suggestion(suggestion):
    lines = suggestion.split('\n')
    category = lines[0].split(':')[1].strip() if ':' in lines[0] else lines[0].strip()
    websites = [line.strip() for line in lines[1:] if line.strip()]
    return category, websites

def web_crawler(category, websites, max_pages=2):
    all_data = []
    print(f"\n--- Crawling category: {category} ---")
    
    for url in websites:
        print(f"Searching in: {url}")
        page = 1
        
        while page <= max_pages:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for item in soup.find_all('div', class_='data-item'):
                    title = item.find('h2').text if item.find('h2') else 'No title'
                    link = item.find('a')['href'] if item.find('a') else 'No link'
                    all_data.append({'title': title, 'link': link, 'source': url})
                
                page += 1
            except Exception as e:
                print(f"Error crawling {url}: {e}")
    
    return all_data

def analyze_data(data_path):
    if not os.path.exists(data_path):
        print("Data file does not exist.")
        return
    
    print("Data file found. Analyzing...")
    df = pd.read_csv(data_path)

    numeric_columns = ['tmax', 'tmin', 'tavg', 'departure', 'HDD', 'CDD', 'precipitation', 'new_snow', 'snow_depth']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].replace('T', '0.0'), errors='coerce')

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y')

    print("\n--- Basic Insights ---")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)

    print("\n--- Advanced Insights ---")
    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nCorrelation Matrix:")
    corr_matrix = df.corr()
    print(corr_matrix)

    return df.head().to_dict()

# Example usage
data_path = "/Users/vishaalchandrasekar/Desktop/nyc_temp.csv"
sample_data = analyze_data(data_path)
dataset_description = f"A dataset with columns: {', '.join(sample_data.keys())}"

category, websites = get_category_and_websites(dataset_description)
crawled_data = web_crawler(category, websites)

print(f"\nCategory suggested by Llama-3.1-8B: {category}")
print("Websites suggested by Llama-3.1-8B:")
for website in websites:
    print(f"- {website}")

print("\nCrawled data sample:")
for item in crawled_data[:5]:
    print(f"Title: {item['title']}, Source: {item['source']}")
