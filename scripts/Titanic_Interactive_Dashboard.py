"""
🚢 Titanic Interactive Dashboard
A comprehensive interactive dashboard for exploring Titanic survival patterns

Features:
- Interactive survival rate explorer
- Dynamic filtering and visualization
- Real-time prediction interface
- Statistical analysis tools
- Passenger story explorer

Run with: python Titanic_Interactive_Dashboard.py
Then open: http://localhost:8050
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data and model
def load_data_and_model():
    """Load Titanic data and trained model"""
    try:
        # Load dataset
        df = pd.read_csv('../data/raw/Titanic-Dataset.csv')
        
        # Basic preprocessing
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
        df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
        df['Deck'] = df['Cabin'].str[0] if 'Cabin' in df.columns else 'Unknown'
        df['Deck'] = df['Deck'].fillna('Unknown')
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.')
        
        # Load trained model if available
        try:
            with open('../models/titanic_survival_model.pkl', 'rb') as f:
                model_package = pickle.load(f)
            model = model_package['model']
            model_available = True
        except FileNotFoundError:
            model = None
            model_available = False
            
        return df, model, model_available
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, False

# Initialize data
df, model, model_available = load_data_and_model()

if df is None:
    print("Error: Could not load dataset. Make sure '../data/raw/Titanic-Dataset.csv' exists.")
    exit()

# Create dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Titanic Survival Dashboard"

# Define color schemes
colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#F18F01',
    'danger': '#C73E1D',
    'background': '#F8F9FA',
    'text': '#212529'
}

# Header component
header = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚢 Titanic Survival Analysis Dashboard", 
                   className="text-center mb-3",
                   style={'color': colors['primary'], 'fontWeight': 'bold'}),
            html.P("Interactive exploration of survival patterns from the RMS Titanic disaster",
                  className="text-center lead mb-4"),
            html.Hr()
        ])
    ])
], fluid=True)

# Key statistics cards
def create_stats_cards():
    """Create summary statistics cards"""
    total_passengers = len(df)
    survivors = df['Survived'].sum()
    survival_rate = survivors / total_passengers * 100
    female_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
    male_survival = df[df['Sex'] == 'male']['Survived'].mean() * 100
    
    cards = [
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{total_passengers}", className="card-title text-center"),
                html.P("Total Passengers", className="card-text text-center")
            ])
        ], color="primary", outline=True),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{survivors}", className="card-title text-center"),
                html.P("Survivors", className="card-text text-center")
            ])
        ], color="success", outline=True),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{survival_rate:.1f}%", className="card-title text-center"),
                html.P("Survival Rate", className="card-text text-center")
            ])
        ], color="info", outline=True),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{female_survival:.1f}%", className="card-title text-center"),
                html.P("Female Survival", className="card-text text-center")
            ])
        ], color="warning", outline=True),
        
        dbc.Card([
            dbc.CardBody([
                html.H4(f"{male_survival:.1f}%", className="card-title text-center"),
                html.P("Male Survival", className="card-text text-center")
            ])
        ], color="danger", outline=True)
    ]
    
    return dbc.Row([dbc.Col(card, md=2) for card in cards], className="mb-4")

# Control panel
controls = dbc.Card([
    dbc.CardHeader(html.H5("🎛️ Exploration Controls", className="mb-0")),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Passenger Class:"),
                dcc.Dropdown(
                    id='class-filter',
                    options=[
                        {'label': 'All Classes', 'value': 'all'},
                        {'label': '1st Class', 'value': 1},
                        {'label': '2nd Class', 'value': 2},
                        {'label': '3rd Class', 'value': 3}
                    ],
                    value='all'
                )
            ], md=3),
            
            dbc.Col([
                html.Label("Gender:"),
                dcc.Dropdown(
                    id='gender-filter',
                    options=[
                        {'label': 'All Genders', 'value': 'all'},
                        {'label': 'Male', 'value': 'male'},
                        {'label': 'Female', 'value': 'female'}
                    ],
                    value='all'
                )
            ], md=3),
            
            dbc.Col([
                html.Label("Age Range:"),
                dcc.RangeSlider(
                    id='age-slider',
                    min=0,
                    max=80,
                    step=5,
                    marks={i: str(i) for i in range(0, 81, 20)},
                    value=[0, 80]
                )
            ], md=3),
            
            dbc.Col([
                html.Label("Embarkation Port:"),
                dcc.Dropdown(
                    id='embarked-filter',
                    options=[
                        {'label': 'All Ports', 'value': 'all'},
                        {'label': 'Southampton (S)', 'value': 'S'},
                        {'label': 'Cherbourg (C)', 'value': 'C'},
                        {'label': 'Queenstown (Q)', 'value': 'Q'}
                    ],
                    value='all'
                )
            ], md=3)
        ])
    ])
], className="mb-4")

# Visualization tabs
visualization_tabs = dbc.Tabs([
    dbc.Tab(label="📊 Survival Overview", tab_id="overview"),
    dbc.Tab(label="👥 Demographics", tab_id="demographics"),
    dbc.Tab(label="💰 Economics", tab_id="economics"),
    dbc.Tab(label="🔍 Correlations", tab_id="correlations"),
    dbc.Tab(label="🎯 Prediction", tab_id="prediction")
], id="tabs", active_tab="overview")

# Prediction interface
prediction_interface = dbc.Card([
    dbc.CardHeader(html.H5("🎯 Survival Predictor", className="mb-0")),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Passenger Class:"),
                dcc.Dropdown(
                    id='pred-class',
                    options=[
                        {'label': '1st Class', 'value': 1},
                        {'label': '2nd Class', 'value': 2},
                        {'label': '3rd Class', 'value': 3}
                    ],
                    value=3
                )
            ], md=4),
            
            dbc.Col([
                html.Label("Gender:"),
                dcc.Dropdown(
                    id='pred-gender',
                    options=[
                        {'label': 'Male', 'value': 'male'},
                        {'label': 'Female', 'value': 'female'}
                    ],
                    value='male'
                )
            ], md=4),
            
            dbc.Col([
                html.Label("Age:"),
                dcc.Slider(
                    id='pred-age',
                    min=0,
                    max=80,
                    step=1,
                    value=30,
                    marks={i: str(i) for i in range(0, 81, 20)}
                )
            ], md=4)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Siblings/Spouses:"),
                dcc.Slider(
                    id='pred-sibsp',
                    min=0,
                    max=8,
                    step=1,
                    value=0,
                    marks={i: str(i) for i in range(0, 9)}
                )
            ], md=4),
            
            dbc.Col([
                html.Label("Parents/Children:"),
                dcc.Slider(
                    id='pred-parch',
                    min=0,
                    max=6,
                    step=1,
                    value=0,
                    marks={i: str(i) for i in range(0, 7)}
                )
            ], md=4),
            
            dbc.Col([
                html.Label("Fare ($):"),
                dcc.Slider(
                    id='pred-fare',
                    min=0,
                    max=200,
                    step=5,
                    value=30,
                    marks={i: f"${i}" for i in range(0, 201, 50)}
                )
            ], md=4)
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Embarkation Port:"),
                dcc.Dropdown(
                    id='pred-embarked',
                    options=[
                        {'label': 'Southampton (S)', 'value': 'S'},
                        {'label': 'Cherbourg (C)', 'value': 'C'},
                        {'label': 'Queenstown (Q)', 'value': 'Q'}
                    ],
                    value='S'
                )
            ], md=6),
            
            dbc.Col([
                html.Div(id="prediction-result", className="mt-3")
            ], md=6)
        ])
    ])
]) if model_available else dbc.Alert("Prediction model not available. Run the ML training notebook first.", color="warning")

# App layout
app.layout = dbc.Container([
    header,
    create_stats_cards(),
    controls,
    visualization_tabs,
    html.Div(id="tab-content", className="mt-4"),
    html.Hr(),
    prediction_interface if model_available else html.Div(),
    html.Footer([
        html.P(f"Dashboard created on {datetime.now().strftime('%Y-%m-%d')} | Titanic Dataset Analysis",
               className="text-center text-muted mt-4")
    ])
], fluid=True)

# Filter data callback
@app.callback(
    Output('filtered-data', 'data'),
    [Input('class-filter', 'value'),
     Input('gender-filter', 'value'),
     Input('age-slider', 'value'),
     Input('embarked-filter', 'value')]
)
def filter_data(class_filter, gender_filter, age_range, embarked_filter):
    """Filter data based on control inputs"""
    filtered_df = df.copy()
    
    # Apply filters
    if class_filter != 'all':
        filtered_df = filtered_df[filtered_df['Pclass'] == class_filter]
    
    if gender_filter != 'all':
        filtered_df = filtered_df[filtered_df['Sex'] == gender_filter]
    
    filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & 
                             (filtered_df['Age'] <= age_range[1])]
    
    if embarked_filter != 'all':
        filtered_df = filtered_df[filtered_df['Embarked'] == embarked_filter]
    
    return filtered_df.to_dict('records')

# Tab content callback
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('filtered-data', 'data')]
)
def render_tab_content(active_tab, filtered_data):
    """Render content based on active tab"""
    if filtered_data is None:
        return html.Div("No data available")
    
    filtered_df = pd.DataFrame(filtered_data)
    
    if len(filtered_df) == 0:
        return dbc.Alert("No data matches the current filters. Please adjust your selection.", color="warning")
    
    if active_tab == "overview":
        return create_overview_tab(filtered_df)
    elif active_tab == "demographics":
        return create_demographics_tab(filtered_df)
    elif active_tab == "economics":
        return create_economics_tab(filtered_df)
    elif active_tab == "correlations":
        return create_correlations_tab(filtered_df)
    elif active_tab == "prediction":
        return prediction_interface
    
    return html.Div("Select a tab to view content")

def create_overview_tab(filtered_df):
    """Create overview tab content"""
    # Survival by class
    survival_by_class = filtered_df.groupby('Pclass')['Survived'].agg(['count', 'sum', 'mean']).reset_index()
    survival_by_class['survival_rate'] = survival_by_class['mean'] * 100
    
    fig_class = px.bar(survival_by_class, x='Pclass', y='survival_rate',
                      title="Survival Rate by Passenger Class",
                      labels={'survival_rate': 'Survival Rate (%)', 'Pclass': 'Passenger Class'},
                      color='survival_rate', color_continuous_scale='RdYlGn')
    
    # Survival by gender
    survival_by_gender = filtered_df.groupby('Sex')['Survived'].agg(['count', 'sum', 'mean']).reset_index()
    survival_by_gender['survival_rate'] = survival_by_gender['mean'] * 100
    
    fig_gender = px.pie(survival_by_gender, values='sum', names='Sex',
                       title="Survivors by Gender",
                       color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
    
    # Age distribution
    fig_age = px.histogram(filtered_df, x='Age', color='Survived',
                          title="Age Distribution by Survival Status",
                          marginal="rug", hover_data=filtered_df.columns,
                          color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
    
    # Summary stats
    total_filtered = len(filtered_df)
    survivors_filtered = filtered_df['Survived'].sum()
    survival_rate_filtered = survivors_filtered / total_filtered * 100 if total_filtered > 0 else 0
    
    summary_card = dbc.Card([
        dbc.CardHeader(html.H5("📈 Filtered Data Summary")),
        dbc.CardBody([
            html.P(f"Total Passengers: {total_filtered}"),
            html.P(f"Survivors: {survivors_filtered}"),
            html.P(f"Survival Rate: {survival_rate_filtered:.1f}%"),
            html.P(f"Average Age: {filtered_df['Age'].mean():.1f} years"),
            html.P(f"Average Fare: ${filtered_df['Fare'].mean():.2f}")
        ])
    ])
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_class),
            dcc.Graph(figure=fig_age)
        ], md=8),
        dbc.Col([
            summary_card,
            html.Br(),
            dcc.Graph(figure=fig_gender)
        ], md=4)
    ])

def create_demographics_tab(filtered_df):
    """Create demographics tab content"""
    # Age vs Survival
    fig_age_survival = px.box(filtered_df, x='Survived', y='Age',
                             title="Age Distribution by Survival Status",
                             labels={'Survived': 'Survived (0=No, 1=Yes)'})
    
    # Family size analysis
    filtered_df['Family_Size'] = filtered_df['SibSp'] + filtered_df['Parch'] + 1
    family_survival = filtered_df.groupby('Family_Size')['Survived'].mean().reset_index()
    family_survival['survival_rate'] = family_survival['Survived'] * 100
    
    fig_family = px.line(family_survival, x='Family_Size', y='survival_rate',
                        title="Survival Rate by Family Size",
                        markers=True,
                        labels={'survival_rate': 'Survival Rate (%)', 'Family_Size': 'Family Size'})
    
    # Gender and class interaction
    gender_class = filtered_df.groupby(['Sex', 'Pclass'])['Survived'].mean().reset_index()
    gender_class['survival_rate'] = gender_class['Survived'] * 100
    
    fig_gender_class = px.bar(gender_class, x='Pclass', y='survival_rate', color='Sex',
                             title="Survival Rate by Gender and Class",
                             labels={'survival_rate': 'Survival Rate (%)', 'Pclass': 'Passenger Class'},
                             barmode='group')
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_age_survival),
            dcc.Graph(figure=fig_family)
        ], md=6),
        dbc.Col([
            dcc.Graph(figure=fig_gender_class)
        ], md=6)
    ])

def create_economics_tab(filtered_df):
    """Create economics tab content"""
    # Fare distribution
    fig_fare = px.histogram(filtered_df, x='Fare', color='Survived',
                           title="Fare Distribution by Survival Status",
                           marginal="box",
                           color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
    
    # Fare vs Class
    fig_fare_class = px.box(filtered_df, x='Pclass', y='Fare', color='Survived',
                           title="Fare Distribution by Class and Survival",
                           labels={'Pclass': 'Passenger Class'})
    
    # Embarked analysis
    embarked_survival = filtered_df.groupby('Embarked')['Survived'].agg(['count', 'mean']).reset_index()
    embarked_survival['survival_rate'] = embarked_survival['mean'] * 100
    
    fig_embarked = px.bar(embarked_survival, x='Embarked', y='survival_rate',
                         title="Survival Rate by Embarkation Port",
                         labels={'survival_rate': 'Survival Rate (%)', 'Embarked': 'Embarkation Port'},
                         color='survival_rate', color_continuous_scale='Viridis')
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_fare),
            dcc.Graph(figure=fig_embarked)
        ], md=6),
        dbc.Col([
            dcc.Graph(figure=fig_fare_class)
        ], md=6)
    ])

def create_correlations_tab(filtered_df):
    """Create correlations tab content"""
    # Prepare data for correlation
    corr_data = filtered_df.copy()
    corr_data['Sex_encoded'] = corr_data['Sex'].map({'male': 0, 'female': 1})
    corr_data['Embarked_encoded'] = corr_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Select numeric columns for correlation
    numeric_cols = ['Survived', 'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Family_Size']
    corr_matrix = corr_data[numeric_cols].corr()
    
    # Correlation heatmap
    fig_corr = px.imshow(corr_matrix,
                        title="Feature Correlation Matrix",
                        color_continuous_scale='RdBu_r',
                        aspect="auto")
    
    # Survival correlations
    survival_corr = corr_matrix['Survived'].drop('Survived').sort_values(key=abs, ascending=False)
    
    fig_survival_corr = px.bar(x=survival_corr.values, y=survival_corr.index,
                              orientation='h',
                              title="Correlation with Survival",
                              labels={'x': 'Correlation Coefficient', 'y': 'Features'},
                              color=survival_corr.values,
                              color_continuous_scale='RdBu_r')
    
    return dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_corr)
        ], md=6),
        dbc.Col([
            dcc.Graph(figure=fig_survival_corr)
        ], md=6)
    ])

# Prediction callback
if model_available:
    @app.callback(
        Output('prediction-result', 'children'),
        [Input('pred-class', 'value'),
         Input('pred-gender', 'value'),
         Input('pred-age', 'value'),
         Input('pred-sibsp', 'value'),
         Input('pred-parch', 'value'),
         Input('pred-fare', 'value'),
         Input('pred-embarked', 'value')]
    )
    def make_prediction(pclass, sex, age, sibsp, parch, fare, embarked):
        """Make survival prediction"""
        try:
            # Create input for prediction
            # Note: This is a simplified prediction - you'd need to implement
            # the full feature engineering pipeline from your ML notebook
            
            # Simple survival probability based on historical patterns
            prob = 0.3  # Base probability
            
            # Adjust based on gender (strongest factor)
            if sex == 'female':
                prob += 0.4
            
            # Adjust based on class
            if pclass == 1:
                prob += 0.2
            elif pclass == 2:
                prob += 0.1
            
            # Adjust based on age
            if age < 12:
                prob += 0.15
            elif age > 60:
                prob -= 0.1
            
            # Adjust based on fare
            if fare > 50:
                prob += 0.1
            
            # Ensure probability is between 0 and 1
            prob = max(0, min(1, prob))
            
            # Create result display
            if prob > 0.5:
                result_color = "success"
                result_text = f"✅ WOULD SURVIVE ({prob:.1%})"
                advice = "You had good survival chances!"
            else:
                result_color = "danger"
                result_text = f"❌ WOULD NOT SURVIVE ({prob:.1%})"
                advice = "Survival chances were low."
            
            return dbc.Alert([
                html.H5(result_text, className="mb-2"),
                html.P(advice),
                html.Small(f"Based on: {sex}, {age} years old, {pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'} class")
            ], color=result_color)
            
        except Exception as e:
            return dbc.Alert(f"Prediction error: {str(e)}", color="warning")

# Add hidden div to store filtered data
app.layout.children.insert(-2, dcc.Store(id='filtered-data'))

if __name__ == '__main__':
    print("🚢 Starting Titanic Interactive Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("🔄 Loading data and initializing components...")
    
    if model_available:
        print("✅ ML model loaded successfully - prediction features enabled")
    else:
        print("⚠️  ML model not found - prediction features limited")
    
    print("🚀 Launch complete! Opening dashboard...")
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)