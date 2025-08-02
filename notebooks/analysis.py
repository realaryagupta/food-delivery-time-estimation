import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Still needed for make_subplots if we want to combine later, but individual functions won't use it directly.
from scipy.stats import probplot

# --- Individual Plot Functions (replacing parts of plotly_numerical_analysis) ---

def create_kde_plot(dataframe, column_name, cat_col=None):
    """
    Generates a KDE plot (using Violin plot with only KDE side) for a numerical column using Plotly.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the numerical column to analyze.
        cat_col (str, optional): A categorical column for hue. Defaults to None.
    """
    fig = go.Figure()

    if cat_col:
        for i, category in enumerate(dataframe[cat_col].unique()):
            subset = dataframe[dataframe[cat_col] == category]
            fig.add_trace(
                go.Violin(
                    x=subset[column_name],
                    name=str(category),
                    line_color='rgba(0,0,0,0)',
                    fillcolor=px.colors.qualitative.Vivid[i % len(px.colors.qualitative.Vivid)],
                    scalemode='count',
                    points=False,
                    box_visible=False,
                    legendgroup=str(category), # Add legend group for toggle
                    showlegend=True, # Show legend when cat_col is present
                )
            )
        fig.update_traces(orientation='h')
    else:
        fig.add_trace(
            go.Violin(
                x=dataframe[column_name],
                fillcolor='#2c7be5',
                line_color='rgba(0,0,0,0)',
                scalemode='count',
                points=False,
                box_visible=False,
                showlegend=False # No need for legend without categories
            )
        )
        fig.update_traces(orientation='h')

    fig.update_xaxes(title_text='Kernel Density Estimate', showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title_text=f'Kernel Density Estimate of {column_name}{f" by {cat_col}" if cat_col else ""}',
        title_x=0.5,
        template='plotly_white',
        height=500,
        width=800
    )
    return fig

def create_boxplot(dataframe, column_name, cat_col=None):
    """
    Generates a Boxplot for a numerical column using Plotly.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the numerical column to analyze.
        cat_col (str, optional): A categorical column for hue. Defaults to None.
    """
    fig = go.Figure()

    if cat_col:
        for i, category in enumerate(dataframe[cat_col].unique()):
            subset = dataframe[dataframe[cat_col] == category]
            fig.add_trace(
                go.Box(
                    x=subset[column_name],
                    name=str(category),
                    marker_color=px.colors.qualitative.Vivid[i % len(px.colors.qualitative.Vivid)],
                    boxpoints=False,
                    legendgroup=str(category), # Add legend group for toggle
                    showlegend=True, # Show legend when cat_col is present
                )
            )
        fig.update_traces(orientation='h')
    else:
        fig.add_trace(
            go.Box(
                x=dataframe[column_name],
                marker_color='#2c7be5',
                boxpoints=False,
                showlegend=False # No need for legend without categories
            )
        )
        fig.update_traces(orientation='h')

    fig.update_xaxes(title_text='Boxplot', showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(visible=False)
    fig.update_layout(
        title_text=f'Boxplot of {column_name}{f" by {cat_col}" if cat_col else ""}',
        title_x=0.5,
        template='plotly_white',
        height=500,
        width=800
    )
    return fig

def create_histogram_with_kde(dataframe, column_name, cat_col=None, bins="auto"):
    """
    Generates a Histogram with KDE overlay for a numerical column using Plotly.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the numerical column to analyze.
        cat_col (str, optional): A categorical column for hue. Defaults to None.
        bins (int or str, optional): Number of bins for the histogram. Defaults to "auto".
    """
    fig = go.Figure()

    if cat_col:
        for i, category in enumerate(dataframe[cat_col].unique()):
            subset = dataframe[dataframe[cat_col] == category]
            color = px.colors.qualitative.Vivid[i % len(px.colors.qualitative.Vivid)]
            fig.add_trace(
                go.Histogram(
                    x=subset[column_name],
                    name=str(category),
                    nbinsx=bins,
                    marker_color=color,
                    opacity=0.7,
                    showlegend=True,
                    histnorm='density',
                    legendgroup=str(category),
                )
            )
            # Overlay KDE
            fig.add_trace(
                go.Violin(
                    x=subset[column_name],
                    name=str(category) + ' KDE',
                    line_color='rgba(0,0,0,0)',
                    fillcolor=color,
                    scalemode='count',
                    points=False,
                    box_visible=False,
                    showlegend=False,
                    legendgroup=str(category),
                )
            )
    else:
        fig.add_trace(
            go.Histogram(
                x=dataframe[column_name],
                nbinsx=bins,
                marker_color='#2c7be5',
                opacity=0.7,
                histnorm='density',
                showlegend=False
            )
        )
        fig.add_trace(
            go.Violin(
                x=dataframe[column_name],
                line_color='rgba(0,0,0,0)',
                fillcolor='#2c7be5',
                scalemode='count',
                points=False,
                box_visible=False,
                showlegend=False
            )
        )

    fig.update_xaxes(title_text='Histogram with KDE', showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(title_text='Density', showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_layout(
        title_text=f'Histogram with KDE of {column_name}{f" by {cat_col}" if cat_col else ""}',
        title_x=0.5,
        template='plotly_white',
        height=600,
        width=900
    )
    return fig

# --- Individual Plot Functions (replacing parts of plotly_numerical_categorical_analysis) ---

def create_numerical_categorical_barplot(dataframe, cat_column, num_column):
    """
    Generates a Barplot showing the mean numerical value by category.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        cat_column (str): The name of the categorical column.
        num_column (str): The name of the numerical column.
    """
    mean_data = dataframe.groupby(cat_column)[num_column].mean().reset_index()
    fig = go.Figure(
        go.Bar(
            x=mean_data[cat_column],
            y=mean_data[num_column],
            marker_color=px.colors.qualitative.Set2[0],
        )
    )
    fig.update_xaxes(title_text=cat_column, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(title_text=f'Mean {num_column}', showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_layout(
        title_text=f'Mean {num_column} by {cat_column} (Barplot)',
        title_x=0.5,
        template='plotly_white',
        height=500,
        width=800
    )
    return fig

def create_numerical_categorical_boxplot(dataframe, cat_column, num_column):
    """
    Generates a Boxplot showing the distribution of a numerical variable by a categorical variable.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        cat_column (str): The name of the categorical column.
        num_column (str): The name of the numerical column.
    """
    fig = go.Figure(
        go.Box(
            x=dataframe[cat_column],
            y=dataframe[num_column],
            marker_color=px.colors.qualitative.Pastel[0],
            boxpoints=False,
        )
    )
    fig.update_xaxes(title_text=cat_column, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(title_text=num_column, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_layout(
        title_text=f'Distribution of {num_column} by {cat_column} (Boxplot)',
        title_x=0.5,
        template='plotly_white',
        height=500,
        width=800
    )
    return fig

def create_numerical_categorical_violin_plot(dataframe, cat_column, num_column):
    """
    Generates a Violin plot showing the distribution of a numerical variable by a categorical variable.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        cat_column (str): The name of the categorical column.
        num_column (str): The name of the numerical column.
    """
    fig = go.Figure(
        go.Violin(
            x=dataframe[cat_column],
            y=dataframe[num_column],
            fillcolor=px.colors.qualitative.Set2[1],
            box_visible=True,
            meanline_visible=True,
        )
    )
    fig.update_xaxes(title_text=cat_column, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(title_text=num_column, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_layout(
        title_text=f'Distribution of {num_column} by {cat_column} (Violin Plot)',
        title_x=0.5,
        template='plotly_white',
        height=500,
        width=800
    )
    return fig

def create_numerical_categorical_stripplot(dataframe, cat_column, num_column):
    """
    Generates a Stripplot showing individual observations of a numerical variable by a categorical variable.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        cat_column (str): The name of the categorical column.
        num_column (str): The name of the numerical column.
    """
    # Use plotly.express.strip for proper jittering and a cleaner stripplot
    fig = px.strip(
        dataframe,
        x=cat_column,
        y=num_column,
        title=f'Individual Observations of {num_column} by {cat_column} (Stripplot)',
        template='plotly_white',
        color_discrete_sequence=[px.colors.qualitative.Plotly[0]], # Consistent color
        stripmode='group', # 'group' mode prevents overplotting by distributing points
        hover_data=[cat_column, num_column] # Add hover data for better interactivity
    )

    fig.update_xaxes(title_text=cat_column, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(title_text=num_column, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_layout(
        title_x=0.5,
        height=500,
        width=800
    )
    return fig

# --- Individual Plot Function (existing plotly_categorical_analysis) ---

def create_categorical_countplot(dataframe, column_name):
    """
    Generates a countplot for a categorical column using Plotly.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the categorical column.
    """
    # Countplot
    fig = px.bar(
        dataframe.groupby(column_name).size().reset_index(name='count'),
        x=column_name,
        y='count',
        color=column_name,
        title=f'Count of Categories in {column_name}',
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_xaxes(title_text=column_name, tickangle=45, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(title_text='Count', showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    fig.update_layout(title_x=0.5, height=500, width=800)
    return fig

# --- Individual Plot Functions (replacing parts of plotly_multivariate_analysis) ---

def create_multivariate_barplot(dataframe, num_column, cat_column_1, cat_column_2):
    """
    Generates a grouped Barplot for multivariate analysis.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        num_column (str): The name of the numerical column.
        cat_column_1 (str): The name of the first categorical column.
        cat_column_2 (str): The name of the second categorical column (for hue/color).
    """
    mean_data = dataframe.groupby([cat_column_1, cat_column_2])[num_column].mean().reset_index()
    fig = go.Figure()
    for i, category2 in enumerate(mean_data[cat_column_2].unique()):
        subset = mean_data[mean_data[cat_column_2] == category2]
        fig.add_trace(
            go.Bar(
                x=subset[cat_column_1],
                y=subset[num_column],
                name=str(category2),
                marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                showlegend=True
            )
        )
    fig.update_xaxes(title_text=cat_column_1, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(title_text=f'Mean {num_column}', showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_layout(
        title_text=f'Mean {num_column} by {cat_column_1} and {cat_column_2} (Barplot)',
        title_x=0.5,
        template='plotly_white',
        barmode='group',
        height=600,
        width=900
    )
    return fig

def create_multivariate_boxplot(dataframe, num_column, cat_column_1, cat_column_2):
    """
    Generates a grouped Boxplot for multivariate analysis.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        num_column (str): The name of the numerical column.
        cat_column_1 (str): The name of the first categorical column.
        cat_column_2 (str): The name of the second categorical column (for hue/color).
    """
    fig = go.Figure()
    for i, category2 in enumerate(dataframe[cat_column_2].unique()):
        subset = dataframe[dataframe[cat_column_2] == category2]
        fig.add_trace(
            go.Box(
                x=subset[cat_column_1],
                y=subset[num_column],
                name=str(category2),
                marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                boxpoints=False,
                showlegend=True
            )
        )
    fig.update_xaxes(title_text=cat_column_1, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(title_text=num_column, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_layout(
        title_text=f'Distribution of {num_column} by {cat_column_1} and {cat_column_2} (Boxplot)',
        title_x=0.5,
        template='plotly_white',
        boxmode='group',
        height=600,
        width=900
    )
    return fig

def create_multivariate_violin_plot(dataframe, num_column, cat_column_1, cat_column_2):
    """
    Generates a grouped Violin plot for multivariate analysis.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        num_column (str): The name of the numerical column.
        cat_column_1 (str): The name of the first categorical column.
        cat_column_2 (str): The name of the second categorical column (for hue/color).
    """
    fig = go.Figure()
    for i, category2 in enumerate(dataframe[cat_column_2].unique()):
        subset = dataframe[dataframe[cat_column_2] == category2]
        fig.add_trace(
            go.Violin(
                x=subset[cat_column_1],
                y=subset[num_column],
                name=str(category2),
                marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                box_visible=True,
                meanline_visible=True,
                points=False,
                showlegend=True,
            )
        )
    fig.update_xaxes(title_text=cat_column_1, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(title_text=num_column, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_layout(
        title_text=f'Distribution of {num_column} by {cat_column_1} and {cat_column_2} (Violin Plot)',
        title_x=0.5,
        template='plotly_white',
        violinmode='group',
        height=600,
        width=900
    )
    return fig

def create_multivariate_stripplot(dataframe, num_column, cat_column_1, cat_column_2):
    """
    Generates a Stripplot for multivariate analysis, colored by the second categorical column.
    Returns the Plotly figure.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        num_column (str): The name of the numerical column.
        cat_column_1 (str): The name of the first categorical column.
        cat_column_2 (str): The name of the second categorical column (for hue/color).
    """
    # Using plotly.express for simplicity with color mapping for stripplot
    fig = px.strip(
        dataframe,
        x=cat_column_1,
        y=num_column,
        color=cat_column_2,
        title=f'Individual Observations of {num_column} by {cat_column_1} and {cat_column_2} (Stripplot)',
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        stripmode='group' # 'group' will prevent overplotting
    )
    fig.update_xaxes(title_text=cat_column_1, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_yaxes(title_text=num_column, showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    fig.update_layout(title_x=0.5, height=600, width=900)
    return fig

# --- Probability Plot Function (existing plotly_probplot) ---

def create_probplot(data_series, title_text='Probability Plot'):
    """
    Generates a probability (Q-Q) plot for a data series to assess normality using Plotly.
    Returns the Plotly figure.

    Args:
        data_series (pd.Series or np.array): The data series to plot.
        title_text (str, optional): The title of the plot. Defaults to 'Probability Plot'.
    """
    # Generate theoretical quantiles and sorted values
    osm, osr = probplot(data_series, dist='norm')

    fig = go.Figure()

    # Add scatter plot for observed vs. theoretical quantiles
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode='markers',
            marker=dict(
                symbol='circle',
                size=8,
                color='#3498db',
                line=dict(color='white', width=1)
            ),
            name='Observations'
        )
    )

    # Add Q-Q line
    fig.add_trace(
        go.Scatter(
            x=[osm[0], osm[-1]],
            y=[osr[0], osr[-1]],
            mode='lines',
            line=dict(color='#e74c3c', width=2),
            name='Q-Q Line'
        )
    )

    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Ordered Values',
        template='plotly_white',
        showlegend=False,
        height=600,
        width=1000,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.2)')
    )

    return fig

# --- Example Usage (for testing the functions independently) ---
if __name__ == "__main__":
    data = pd.read_csv('../data/interim/clean_train.csv')

    print("Generating individual figures for demonstration. Uncomment .show() to display them.")

    # Numerical Analysis Plots
    print("\n--- Numerical Analysis Plots ---")
    fig_kde = create_kde_plot(data, 'time', cat_col='traffic')
    # fig_kde.show()
    print("KDE Plot generated.")

    fig_boxplot_num = create_boxplot(data, 'time', cat_col='traffic')
    # fig_boxplot_num.show()
    print("Numerical Boxplot generated.")

    fig_hist_kde = create_histogram_with_kde(data, 'time', cat_col='traffic', bins=20)
    # fig_hist_kde.show()
    print("Histogram with KDE generated.")

    # Numerical-Categorical Analysis Plots
    print("\n--- Numerical-Categorical Analysis Plots ---")
    fig_bar_num_cat = create_numerical_categorical_barplot(data, "day_of_week", "time")
    # fig_bar_num_cat.show()
    print("Numerical-Categorical Barplot generated.")

    fig_boxplot_num_cat = create_numerical_categorical_boxplot(data, "day_of_week", "time")
    # fig_boxplot_num_cat.show()
    print("Numerical-Categorical Boxplot generated.")

    fig_violin_num_cat = create_numerical_categorical_violin_plot(data, "day_of_week", "time")
    # fig_violin_num_cat.show()
    print("Numerical-Categorical Violin Plot generated.")

    fig_strip_num_cat = create_numerical_categorical_stripplot(data, "day_of_week", "time")
    # fig_strip_num_cat.show()
    print("Numerical-Categorical Stripplot generated.")

    # Categorical Analysis Plot
    print("\n--- Categorical Analysis Plot ---")
    fig_countplot_cat = create_categorical_countplot(data, 'traffic')
    # fig_countplot_cat.show()
    print("Categorical Countplot generated.")

    # Multivariate Analysis Plots
    print("\n--- Multivariate Analysis Plots ---")
    fig_mv_barplot = create_multivariate_barplot(data, "time", "day_of_week", "order_type")
    # fig_mv_barplot.show()
    print("Multivariate Barplot generated.")

    fig_mv_boxplot = create_multivariate_boxplot(data, "time", "day_of_week", "order_type")
    # fig_mv_boxplot.show()
    print("Multivariate Boxplot generated.")

    fig_mv_violin = create_multivariate_violin_plot(data, "time", "day_of_week", "order_type")
    # fig_mv_violin.show()
    print("Multivariate Violin Plot generated.")

    fig_mv_stripplot = create_multivariate_stripplot(data, "time", "day_of_week", "order_type")
    # fig_mv_stripplot.show()
    print("Multivariate Stripplot generated.")

    # Probability Plot
    print("\n--- Probability Plot ---")
    fig_prob = create_probplot(data['time'], title_text='Probability Plot for Delivery Time')
    # fig_prob.show()
    print("Probability Plot generated.")

    print("\nAll individual Plotly figures have been generated and are ready to be returned or displayed.")