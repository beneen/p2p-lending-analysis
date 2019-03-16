import pandas as pd
from sklearn import preprocessing
import sys

"""
-----
Loading in the accepted.csv file and pre-processing the data for analysis
-----
"""

def map_values_to_int(feature_selected_dataset):
    #mapping text values to int for our analysis
    feature_selected_dataset['grade'] = feature_selected_dataset['grade'].map({'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1})
    feature_selected_dataset["home_ownership"] = feature_selected_dataset["home_ownership"].map(
        {"MORTGAGE": 6, "RENT": 5, "OWN": 4, "OTHER": 3, "NONE": 2, "ANY": 1})
    feature_selected_dataset["emp_length"] = feature_selected_dataset["emp_length"].replace(
        {'years': '', 'year': '', ' ': '', '<': '', '\+': '', 'n/a': '0'}, regex=True)
    feature_selected_dataset["emp_length"] = feature_selected_dataset["emp_length"].apply(lambda x: int(x))

    feature_selected_dataset.fillna(feature_selected_dataset.mean(), inplace=True)

    return feature_selected_dataset

def select_features_for_analysis(reduced_column_dataset):
    #getting the features we want
    features_for_analysis = ['funded_amnt', 'emp_length', 'annual_inc', 'home_ownership', 'grade',
                "last_pymnt_amnt", "mort_acc", "pub_rec", "int_rate", "open_acc", "num_actv_rev_tl",
                "mo_sin_rcnt_rev_tl_op", "mo_sin_old_rev_tl_op", "bc_util", "bc_open_to_buy",
                "avg_cur_bal", "acc_open_past_24mths",
                'loan_status']

    return reduced_column_dataset[features_for_analysis]

def strip_out_specified_columns(original_dataset_with_loan_status):
    #getting rid of columns that we dont need from our dataset. This should aid with performance
    column_names_to_delete = ["delinq_2yrs", "last_pymnt_d", "chargeoff_within_12_mths", "delinq_amnt", "emp_title", "term",
                     "emp_title", "pymnt_plan", "purpose", "title", "zip_code", "verification_status", "dti",
                     "earliest_cr_line", "initial_list_status", "out_prncp",
                     "pymnt_plan", "num_tl_90g_dpd_24m", "num_tl_30dpd", "num_tl_120dpd_2m", "num_accts_ever_120_pd",
                     "delinq_amnt",
                     "chargeoff_within_12_mths", "total_rec_late_fee", "out_prncp_inv",
                     "issue_d"]


    column_names_to_keep = list(set(column_names_to_delete).symmetric_difference(set(original_dataset_with_loan_status.columns)))
    stripped_dataset = original_dataset_with_loan_status[column_names_to_keep]#
    return stripped_dataset


def boolean_loan_status(original_uncleaned_dataset):
    #converting the loan status column to boolean from pre-defined text for our analysis
    loan_status_segment = original_uncleaned_dataset[
        (original_uncleaned_dataset['loan_status'] == "Fully Paid") | (original_uncleaned_dataset['loan_status'] == "Charged Off")]
    loan_status_to_boolean_dict = {"Fully Paid": 0, "Charged Off": 1}
    boolean_loan_status_dataset = loan_status_segment.replace({"loan_status": loan_status_to_boolean_dict})
    return boolean_loan_status_dataset


def load_dataset():
    #loading dataset
    try:
        #hardcoded, need to change this
        original_uncleaned_dataset = pd.read_csv(r'/home/beneen/PycharmProjects/p2p_lending/analysis/data_sample/accepted.csv')
    except:
        print("Dataset file not found")
        sys.exit(1)

    return original_uncleaned_dataset


def dataset_clean():
    #loading in the dataset and pre-processing ready for analysis
    original_uncleaned_dataset = load_dataset()
    boolean_loan_status_dataset = boolean_loan_status(original_uncleaned_dataset)
    original_dataset_with_loan_status = boolean_loan_status_dataset.dropna(thresh=340000, axis=1)
    reduced_column_dataset = strip_out_specified_columns(original_dataset_with_loan_status)
    feature_selected_dataset = select_features_for_analysis(reduced_column_dataset)

    #forward fill and drop na
    feature_selected_dataset = feature_selected_dataset.fillna(method = 'ffill').dropna()


    #mapping more values to int for analysis
    mapped_values_to_int_dataset = map_values_to_int(feature_selected_dataset)

    #standard scaling
    scaler = preprocessing.StandardScaler()
    fields = mapped_values_to_int_dataset.columns.values[:-1]
    scaled_dataset = pd.DataFrame(scaler.fit_transform(mapped_values_to_int_dataset[fields]), columns=fields)
    scaled_dataset['loan_status'] = mapped_values_to_int_dataset['loan_status']
    scaled_dataset['loan_status'].value_counts()

    #resetting the loan status column
    loanstatus_zero = scaled_dataset[scaled_dataset["loan_status"] == 0]
    loanstatus_one = scaled_dataset[scaled_dataset["loan_status"] == 1]
    subset_of_loanstatus_0 = loanstatus_zero.sample(n=5500)
    subset_of_loanstatus_1 = loanstatus_one.sample(n=5500)
    scaled_dataset = pd.concat([subset_of_loanstatus_1, subset_of_loanstatus_0])
    scaled_dataset = scaled_dataset.sample(frac=1).reset_index(drop=True)

    return scaled_dataset

def main():
    dataset_clean()

if __name__ == "__main__":
    main()

