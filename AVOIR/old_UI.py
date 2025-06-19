from typing import ClassVar
import streamlit as st
from stqdm import stqdm
from dsl import spec, SpecTracker
import seaborn as sns
import matplotlib.pyplot as plt
from dsl.grammar import (
    create_variable as V,
    Expectation as E,
    RETURN_VARIABLE as r
)
from dsl.grammar import Specification
from dsl.visualization import add_vega_chart, create_viz_spec
from dsl.tests.datasets import get_dataset_names, get_dataset_attr_list, get_dataset_attr_dict, Dataset, get_dataset
from dsl.tests.models import get_models, get_model, ModelBasedTest, InvalidTargetError, InvalidTrainError, _view_maintenance
import pdb
import time

class CustomDatasetModel(ModelBasedTest):
    def __init__(self, dataset, output_var, input_vars, attrs, model: ClassVar[ModelBasedTest]):
        self.output_var = output_var
        self.input_vars = input_vars
        self.attrs = attrs
        self.dataset: Dataset = Dataset(dataset.train, dataset.test, attrs, target=output_var, inputs=input_vars)

        self.model = model(self.dataset)

    def update_spec(self, new_spec):
        if not isinstance(new_spec, Specification):
            raise ValueError("Input provided was not parsed into a specification")

        @spec(new_spec, include_confidence=True)
        def decision_func(**kwargs):
            x = kwargs.get("x")
            # prediction = self.model.model.predict(x.reshape(1, -1)) # TODO??
            prediction = self.model.predict(x.reshape(1, -1))
            return prediction[0]
        self.f_x = decision_func

    @property
    def data(self):
        test_data = []
        for _, d in self.dataset.test.iterrows():
            x = d.loc[self.dataset.inputs].to_numpy()
            data_dict = {
                "x": x
            }
            for ind, attr in enumerate(self.dataset.attributes_dict.keys()):
                data_dict[attr] = d[attr]
            test_data.append(data_dict)
        return test_data


def provide_model_eval_interface(model_obj):
    selected_fairness = st.session_state.get("selected_fairness", "N/A")
    dataset_name = st.session_state.get("selected_dataset_name", "N/A")
    model_obj.update_spec(eval(spec_code))  # TODO DANGEROUS
    if st.button("Run spec analysis"):
        with st.spinner("Running analysis"):
            start_time = time.perf_counter()
            model_obj.run_eval_loop(progress_bar=stqdm)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            st.success(
                f"Fairness metric '{selected_fairness}' took "
                f"{elapsed_time:.4f} seconds to run on dataset '{dataset_name}'.  \n" 
                f"To see the values in the **Top Chart**, you should **double-click** on the desired node and as for the **Bottom Chart**, you should **hold shift and single-click** on the node you want."
            )
        with st.spinner("Generating spec chart"):
            data_values = model_obj.get_tabular_rep()
            viz_spec = create_viz_spec(data_values)
            add_vega_chart(viz_spec)

    else:
        st.markdown("### Press the button to run")

if __name__ == "__main__":
    st.set_page_config(layout='wide')
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
    }
    """, unsafe_allow_html=True)
    
    # datasets
    datasets = get_dataset_names()
    st.header("Dataset Selection")
    selected_dataset_name = st.selectbox("", datasets, index=1)
    selected_dataset = get_dataset(selected_dataset_name)

    dataset_attr_dict = get_dataset_attr_dict(selected_dataset_name)
    st.header("Dataset Attributes")
    st.json(dataset_attr_dict, expanded= False)
    attr_list = selected_dataset.attributes
    #st.write(attr_list)

    # After dataset is selected
    if selected_dataset_name == "Compas":
        # Load the dataset into a DataFrame 
        dataset_df = selected_dataset.train  # Or select test data, or both
        
        # Reshape the data to long format to combine race columns
        race_columns = ["race_African-American", "race_Asian", "race_Caucasian", 
                        "race_Hispanic", "race_Native American", "race_Other"]
        
        # Melt the dataset to have a column for race and a binary value for each row
        melted_df = dataset_df[race_columns].melt(var_name="race", value_name="is_member")
        
        # Filter out rows where `is_member` is False (only keep rows where the individual belongs to that race)
        melted_df = melted_df[melted_df["is_member"] == 1]
        
        # Add a button to trigger the plot display
        if st.button("Show Race Distribution Histogram"):
            st.markdown("### Histogram of Race Distribution")
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            sns.countplot(data=melted_df, x="race")
            
            # Display the plot in Streamlit
            st.pyplot(plt)
    elif selected_dataset_name == "Adult Income":
        # Load the dataset into a DataFrame
        dataset_df = selected_dataset.train  # Or select test data, or both
        
        # Reshape the data to long format to combine race columns
        race_columns = ["sex_Male", "sex_Female"]
        
        # Melt the dataset to have a column for race and a binary value for each row
        melted_df = dataset_df[race_columns].melt(var_name="sex", value_name="is_member")
        
        # Filter out rows where `is_member` is False (only keep rows where the individual belongs to that race)
        melted_df = melted_df[melted_df["is_member"] == 1]
        
        # Add a button to trigger the plot display
        if st.button("Show Gender Distribution Histogram"):
            st.markdown("### Histogram of Gender Distribution")
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            sns.countplot(data=melted_df, x="sex")
            
            # Display the plot in Streamlit
            st.pyplot(plt)

    st.header("Model Selection")
    selected_model_name = st.selectbox("", get_models())
    selected_model = get_model(selected_model_name)

    if selected_model_name == _view_maintenance:
        output_var = selected_dataset.target
        input_vars = attr_list
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Output Variable")
        with col2:
            st.subheader("Input Variables")
        output_var = col1.selectbox("", attr_list, index=attr_list.index(selected_dataset.target))
        input_vars = col2.multiselect("", attr_list, default=selected_dataset.inputs)
        if output_var in input_vars:
            st.markdown('<p class="big-font">NOTE: input includes output</p>', unsafe_allow_html=True)
    
    #st.markdown(f"These attributes can be used as variables in the spec: {', '.join(input_vars)} ")
    st.markdown('<p class="big-font">'f"These attributes can be used as variables in the spec: {', '.join(attr_list)} "'</p>', unsafe_allow_html= True)
    st.markdown('<p class="big-font">Note: Return variable is *r*, basically, the output variable of the model is represented by *r*; and other variables must be input in quotes. *x* refers to the full input</p>', unsafe_allow_html=True)

    test_specs = {
        "Adult Income": 'E(r, given=(V("sex_Male") == 1)) / E(r, given=(V("sex_Female") == 1)) < 1.2',
        "Boston Housing Prices": 'E(r, given=(V("lstat") > 12)) / E(r, given=(V("lstat") < 12)) > 0.6',
        "Rate My Professors": 'E(r, given=(V("gender_male") == 1) & (V("male_dominated_department") == 1)) / E(r, given=(V("gender_female") == 1) & (V("male_dominated_department") == 1)) < 1.2',
        "Compas": '(E(V("two_year_recid"),given=(V("score_text_Low") == 0) & (V("race_African-American") == 1))) / (E(V("two_year_recid"),given=(V("score_text_Low") == 0) & (V("race_Caucasian") == 1))) > 1.0',
        "ACSIncome": 'E(r, given=(V("SEX_1.0") == 1)) / E(r, given=(V("SEX_2.0") == 1)) < 1.2'
    }
    # initial_spec = test_specs.get(selected_dataset_name, test_specs.get("Boston Housing Prices"))
    # spec_code = st.text_input("Spec:", initial_spec)

    # Fairness specifications template dictionary
    fairness_specs_templates = {
        "Demographic Parity": 'E(r, given=(V("{group1_attr}") == {group1_value})) / E(r, given=(V("{group2_attr}") == {group2_value})) < {threshold}',
        "Equalized Odds": '(E(r, given=(V("{group1_attr}") == {group1_value}) & (V("{label_attr}") == {label_value})) / E(r, given=(V("{group2_attr}") == {group2_value}) & (V("{label_attr}") == {label_value})) < {threshold}) & (E(r, given=(V("{group1_attr}") == {group1_value}) & (V("{label_attr}") == {label_value_not})) / E(r, given=(V("{group2_attr}") == {group2_value}) & (V("{label_attr}") == {label_value_not})) < {threshold})' ,
        "Equal Opportunity": 'E(r, given=(V("{group1_attr}") == {group1_value}) & (V("{label_attr}") == {label_value})) / E(r, given=(V("{group2_attr}") == {group2_value}) & (V("{label_attr}") == {label_value})) > {threshold}',
    }

    # Dataset-specific attribute mapping
    dataset_attributes = {
        "Adult Income": {
            "group1_attr": "sex_Male",
            "group1_value": 1,
            "group2_attr": "sex_Female",
            "group2_value": 1,
            "label_attr": "high_income",
            "label_value": 1,
            "label_value_not": 0
        },
        "ACSIncome": {
            "group1_attr": "SEX_1.0",
            "group1_value": 1,
            "group2_attr": "SEX_2.0",
            "group2_value": 1,
            "label_attr": "target",
            "label_value": 1,
            "label_value_not": 0
        },
        "Rate My Professors": {
            "group1_attr": "gender_male",
            "group1_value": 1,
            "group2_attr": "gender_female",
            "group2_value": 1,
            "label_attr": "male_dominated_department",
            "label_value": 1,
        },
        "Compas": {
            "group1_attr": "race_African-American",
            "group1_value": 1,
            "group2_attr": "race_Caucasian",
            "group2_value": 1,
            "label_attr": "two_year_recid",
            "label_value": 1,
            "label_value_not": 0
        },
    }

    # Predefined set of racial groups for COMPAS
    racial_groups = [
        "race_African-American",
        "race_Asian",
        "race_Caucasian",
        "race_Hispanic",
        "race_Native American",
        "race_Other",
    ]

    # Get the attributes for the selected dataset
    selected_dataset_attrs = dataset_attributes.get(selected_dataset_name, {})

    # Display Fairness Notions
    st.header("Fairness Metric Selection")
    selected_fairness = st.selectbox("", list(fairness_specs_templates.keys()))

    st.session_state.selected_fairness = selected_fairness  
    st.session_state.selected_dataset_name = selected_dataset_name  

    # Display threshold selection
    st.markdown('<p class="big-font">Choose the Fairness threshold (e.g., 1.2 for Demographic Parity):</p>', unsafe_allow_html=True)
    threshold = st.slider("", 0.0, 2.0, 1.2)
    st.session_state.threshold = threshold

    # Customization for racial groups in COMPAS dataset only
    if selected_dataset_name == "Compas":
        st.subheader("Customize Race Groups")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Group 1 Race")
        with col2:
            st.subheader("Group 2 Race")
        group1_attr = col1.selectbox("", racial_groups, index=racial_groups.index("race_African-American"))
        group2_attr = col2.selectbox("", racial_groups, index=racial_groups.index("race_Caucasian"))

        # Update dataset-specific attributes with selected races
        selected_dataset_attrs["group1_attr"] = group1_attr
        selected_dataset_attrs["group2_attr"] = group2_attr
        selected_dataset_attrs["group1_value"] = 1  # Fixed value
        selected_dataset_attrs["group2_value"] = 1  # Fixed value

    # Generate the specification dynamically
    if selected_dataset_attrs:
        selected_dataset_attrs["threshold"] = threshold  # Add the threshold to the dataset attributes
        selected_spec_template = fairness_specs_templates[selected_fairness]
        selected_spec_code = selected_spec_template.format(**selected_dataset_attrs)
        st.markdown(f"#### Generated Specification: `{selected_spec_code}`")
    else:
        st.warning("This dataset does not have predefined fairness attributes. Please configure attributes manually.")

    if selected_fairness == "Equalized Odds":
        st.markdown('<p class="big-font">Note: This fairness criterion requires that the model error rates (both false positives and false negatives) are equal for different groups.</p>', unsafe_allow_html=True)
    else:
        if selected_dataset_name == "Compas":
            if selected_fairness == "Demographic Parity":
                st.markdown('<p class="big-font">Note: This metric compares the expected outcomes between two racial groups. It checks if the ratio of the expected outcome for race1 to that of race2 is less than threshold. A ratio near to 1, suggesting the model is fairer in its predictions across races. By evaluating this metric, you can ensure that the model does not disproportionately favor one racial group over another.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="big-font">Note: This metric compares the expected outcomes between race1 and race2 individuals who both have a two-year recidivism risk score of 1. A ratio near to 1 suggests that the model produces more equitable predictions for individuals from both racial groups who share similar recidivism risk profiles. By evaluating this metric, you can ensure that the model does not disproportionately favor one racial group over another, particularly in the context of recidivism prediction.</p>', unsafe_allow_html=True)
        elif selected_dataset_name == "Adult Income":
            if selected_fairness == "Demographic Parity":
                st.markdown('<p class="big-font">Note: This metric compares the expected outcomes between two gender groups. It checks if the ratio of the expected outcome for males to that of females is less than threshold. A ratio near to 1, suggesting the model is fairer in its predictions across genders. By evaluating this metric, you can ensure that the model does not disproportionately favor one gender group over another.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="big-font">Note: This metric compares the expected outcomes between males and females individuals who both have a high income. A ratio near to 1 suggests that the model produces more equitable predictions for individuals from both gender groups who share similar income level. By evaluating this metric, you can ensure that the model does not disproportionately favor one gendr group over another, particularly in the context of high income.</p>', unsafe_allow_html=True)
    # Store the selected specification in session state
    if "selected_spec_code" not in st.session_state or st.session_state.selected_spec_code != selected_spec_code:
        st.session_state.selected_spec_code = selected_spec_code


    # Pass the specification code to the model evaluation
    spec_code = st.session_state.selected_spec_code


    # Store selections in session state to detect changes
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = selected_model_name
    if "input_vars" not in st.session_state:
        st.session_state.input_vars = input_vars
    if "output_var" not in st.session_state:
        st.session_state.output_var = output_var
    if "model_obj" not in st.session_state or \
       st.session_state.selected_model_name != selected_model_name or \
       st.session_state.input_vars != input_vars or \
       st.session_state.output_var != output_var:
        
        # Update session state
        st.session_state.selected_model_name = selected_model_name
        st.session_state.input_vars = input_vars
        st.session_state.output_var = output_var

        try:
            with st.spinner("Training model..."):
                st.session_state.model_obj = CustomDatasetModel(selected_dataset, output_var, input_vars, attr_list, selected_model)
        except InvalidTargetError:
            st.markdown("Oops! Looks like the chosen output variable isn't compatible with the selected model. Select another configuration")
        except InvalidTrainError:
            st.markdown("Oops! Looks like the chosen training data isn't compatible with the selected model. Select another configuration")
    
    # Retrieve the model object from session state and use it
    model_obj = st.session_state.model_obj
    provide_model_eval_interface(model_obj)

    # try:
    #     with st.spinner("Training model..."):
    #         model_obj = CustomDatasetModel(selected_dataset, output_var, input_vars, attr_list, selected_model)
    #     provide_model_eval_interface(model_obj)
    # except InvalidTargetError:
    #     st.markdown("Oops! Looks like the chosen output variable isn't compatible with the selected model. Select another configuration")
    # except InvalidTrainError:
    #     st.markdown("Oops! Looks like the chosen training data isn't compatible with the selected model. Select another configuration")