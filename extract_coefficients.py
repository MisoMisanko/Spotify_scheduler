import joblib
import pandas as pd
import numpy as np

def extract_model_coefficients(model_file='models/production_personality_models.pkl'):
    """
    Extract and analyze Ridge regression coefficients from trained models
    """
    
    # Load the trained models
    try:
        model_data = joblib.load(model_file)
        models = model_data['models']
        feature_names = model_data['feature_names']
        print(f"Loaded models for: {list(models.keys())}")
        print(f"Features: {feature_names}\n")
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
    
    # Extract coefficients for each trait
    coefficient_analysis = {}
    
    for trait, model in models.items():
        print(f"=== {trait.upper()} COEFFICIENTS ===")
        
        # Get Ridge regression coefficients from the pipeline
        # The pipeline has: StandardScaler -> Ridge
        ridge_model = model.named_steps['model']  # This is the Ridge regression
        coefficients = ridge_model.coef_
        
        # Create coefficient dictionary
        coef_dict = dict(zip(feature_names, coefficients))
        
        # Sort by absolute coefficient value (most influential first)
        sorted_coeffs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Separate positive and negative coefficients
        positive_coeffs = [(genre, coef) for genre, coef in sorted_coeffs if coef > 0]
        negative_coeffs = [(genre, coef) for genre, coef in sorted_coeffs if coef < 0]
        
        print("\nPositive Coefficients (increasing trait):")
        for genre, coef in positive_coeffs[:10]:  # Top 10
            print(f"  {genre}: +{coef:.3f}")
        
        print("\nNegative Coefficients (decreasing trait):")
        for genre, coef in negative_coeffs[:10]:  # Top 10
            print(f"  {genre}: {coef:.3f}")
        
        # Store for analysis
        coefficient_analysis[trait] = {
            'all_coefficients': coef_dict,
            'positive': positive_coeffs,
            'negative': negative_coeffs,
            'model_performance': model_data['performance_metrics'][trait]
        }
        
        print(f"\nModel Performance:")
        print(f"  Test R²: {model_data['performance_metrics'][trait]['test_r2']:.3f}")
        print(f"  Cross-val R²: {model_data['performance_metrics'][trait]['cv_score']:.3f}")
        print("-" * 50)
    
    return coefficient_analysis

def create_coefficient_table(coefficient_analysis):
    """
    Create a formatted table of all coefficients
    """
    
    # Create DataFrame with all coefficients
    coef_data = []
    
    for trait, data in coefficient_analysis.items():
        for genre, coef in data['all_coefficients'].items():
            coef_data.append({
                'Trait': trait,
                'Genre': genre,
                'Coefficient': coef,
                'Abs_Coefficient': abs(coef),
                'Direction': 'Positive' if coef > 0 else 'Negative'
            })
    
    df = pd.DataFrame(coef_data)
    
    # Pivot table for easy viewing
    pivot_table = df.pivot(index='Genre', columns='Trait', values='Coefficient')
    
    print("\n=== COMPLETE COEFFICIENT MATRIX ===")
    print(pivot_table.round(3))
    
    return df, pivot_table

def analyze_openness_specifically(coefficient_analysis):
    """
    Focus on Openness since it had the best performance
    """
    if 'Openness' not in coefficient_analysis:
        print("Openness model not found!")
        return
    
    openness_data = coefficient_analysis['Openness']
    
    print("\n=== DETAILED OPENNESS ANALYSIS ===")
    print(f"Model Performance: R² = {openness_data['model_performance']['test_r2']:.3f}")
    
    print("\nTop 10 Positive Predictors (Higher Openness):")
    for i, (genre, coef) in enumerate(openness_data['positive'][:10], 1):
        print(f"{i:2d}. {genre:<20} (+{coef:.3f})")
    
    print("\nTop 10 Negative Predictors (Lower Openness):")
    for i, (genre, coef) in enumerate(openness_data['negative'][:10], 1):
        print(f"{i:2d}. {genre:<20} ({coef:.3f})")
    
    # These are the numbers that should go in your dissertation!
    print("\n=== FOR DISSERTATION (Section 4.3) ===")
    print("Openness Predictors (R² = 0.36):")
    print("\nPositive coefficients:")
    for genre, coef in openness_data['positive'][:5]:
        print(f"* {genre} (+{coef:.2f})")
    
    print("\nNegative coefficients:")
    for genre, coef in openness_data['negative'][:5]:
        print(f"* {genre} ({coef:.2f})")

def main():
    """
    Run complete coefficient extraction and analysis
    """
    print("EXTRACTING RIDGE REGRESSION COEFFICIENTS")
    print("=" * 60)
    
    # Extract coefficients
    coefficient_analysis = extract_model_coefficients()
    
    if coefficient_analysis is None:
        print("Failed to load model data!")
        return
    
    # Create coefficient table
    df, pivot_table = create_coefficient_table(coefficient_analysis)
    
    # Analyze Openness specifically
    analyze_openness_specifically(coefficient_analysis)
    
    # Save results
    try:
        df.to_csv('coefficient_analysis.csv', index=False)
        pivot_table.to_csv('coefficient_matrix.csv')
        print(f"\n✅ Results saved to coefficient_analysis.csv and coefficient_matrix.csv")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return coefficient_analysis

if __name__ == "__main__":
    coefficients = main()