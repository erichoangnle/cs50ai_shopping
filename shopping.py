import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Create month dict for mapping
    month = {
        'Jan': 0,
        'Feb': 1,
        'Mar': 2,
        'Apr': 3,
        'May': 4,
        'June': 5,
        'Jul': 6,
        'Aug': 7,
        'Sep': 8,
        'Oct': 9,
        'Nov': 10,
        'Dec': 11
    }

    # Initialize evidence and labels lists
    evidence = []
    labels = []

    # Open csv file for reading
    with open(filename) as file:
        reader = csv.reader(file)
        next(reader) # Skip first line
        for line in reader:
            # Add labels to list
            labels.append(0 if line[-1] == 'FALSE' else 1) 
            # Delete label from line
            del line[-1]
            # Add evidence to list
            evidence.append(line)
            
    # Manipulate items in evidence
    for item in evidence:
        item[10] = int(month[item[10]])
        item[0], item[2], item[4], item[11] = map(int, (item[0], item[2], item[4], item[11]))
        item[12], item[13], item[14] = map(int, (item[12], item[13], item[14]))
        item[1], item[3], item[5], item[6], item[7], item[8], item[9] = map(float, (item[1], item[3], item[5], item[6], item[7], item[8], item[9]))
        item[15] = 1 if item[15] == 'Returning_Visitor' else 0
        item[16] = 1 if item[16] == 'TRUE' else 0

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Choose model type
    model = KNeighborsClassifier(n_neighbors=1)
    # Fit model
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Initialize count variables
    positive_count = 0
    negative_count = 0
    positive_correct = 0
    negative_correct = 0

    # Loop over labels and predictions
    for i in range(len(labels)):
        if labels[i] == 1:
            positive_count += 1
            if predictions[i] == 1:
                positive_correct += 1
        else:
            negative_count += 1
            if predictions[i] == 0:
                negative_correct += 1

    # Calculate sensitivity and specificity
    sensitivity = positive_correct / positive_count
    specificity = negative_correct / negative_count

    return sensitivity, specificity


if __name__ == "__main__":
    main()
