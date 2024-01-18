import numpy as np
import math

## Aggregation algorithms
# 1. Base Method
# 2. Dempster Shafer theory of evidence
# 3. Uninorm Aggregation
# 4. Z-score
# 5. Weighted Average

# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def get_single_emotion_score(scores, distance):
    data = np.array(scores)
    if data.std() > 0.0:
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        # Marking outliers based on z-scores
        not_outliers = np.where((z_scores <= distance))  # Example condition for outliers (modify as needed)
        # Highlighting outliers with a different color and marker

        result = [data[i] for i in list(not_outliers[0])]
        return np.mean(result)
    return data.mean()
    

def get_single_emotion_score_notclear(scores):
    data = np.array(scores)
    # dataset  = data
    # print(dataset)

    # sm=0
    # for i in range(len(dataset)):
    #     sm+=dataset[i]
    # mean = sm/len(dataset)

    # deviation_sum = 0
    # for i in range(len(dataset)):
    #     deviation_sum+=(dataset[i]- mean)**2

    # psd = math.sqrt((deviation_sum)/len(dataset))
    # # print("SD:", psd)

    if data.std() > 0.0:
        zscores = ((data - data.mean())/data.std())
    else:
        return data.mean()
    
    print(zscores)

    # normalized_dataset = (new_data - np.min(new_data)) / (np.max(new_data) - np.min(new_data))
    # # print(normalized_dataset)
    # return normalized_dataset.mean()


def inverse(d):
    # Create a NumPy array with the number
    array = np.array(d)

    # Calculate the reciprocal of the number
    reciprocal = np.reciprocal(array)
    return list(reciprocal)

def get_new_emotion_score(scores):
    data = np.array(scores)
    median = np.median(data)
    # print("@median1: ", median)
    # print("@scores: ", scores)
    dis = []
    for i in scores:
        # print("median2: ", median)
        # print("i: ", i)
        # print("Distance: ", abs(median - i))
        if abs(median-  i) > 0.0:
            dis.append(abs(median - i))
        else:
            dis.append(0.0001)

    # print(dis)
    lst = inverse(dis)
    sum_val = sum(lst)
    # print(lst)
    # Print the reciprocal of the number
    # print(sum_val)

    S = []
    for (i,j) in zip(lst, scores):
        # inv = inverse([i])
        s = (i*j)/sum_val
        S.append(s)

    # dataset = np.array(S)
    # print(sum(S))
    return sum(S)


def base_method(Sa, num_labels):
    A = [0]*num_labels
    for i in range(num_labels):
        for j in range(len(Sa)):
            A[i] += Sa[j][i]

    result = softmax(A)
    return result

def method_with_std_zscore(Sa, num_labels, distance):
    A = [0]*num_labels
    for i in range(num_labels):
        s = []
        for j in range(len(Sa)):
            s.append(Sa[j][i])
        # print("S: ", s)
        A[i] = get_single_emotion_score(s, distance)
    # print("A: ", A)

    result = softmax(A)
    return result

def method3(Sa, num_labels):
    A = [0]*num_labels
    for i in range(num_labels):
        s = []
        for j in range(len(Sa)):
            s.append(Sa[j][i])
        # print("S: ", s)
        A[i] = get_new_emotion_score(s)
    return A

def get_zcore_weighted_avg(scores):
    data = np.array(scores)

    z_scores = []
    if data.std() > 0.0:
        z_scores = list(np.abs((data - np.mean(data)) / np.std(data)))
    else:
        for i in range(len(data)):
            z_scores.append(0.0001)
        # z_scores = np.array(z_scores)

    # print(dis)
    lst = inverse(z_scores)
    sum_val = sum(lst)
    # print(lst)
    # Print the reciprocal of the number
    # print(sum_val)

    S = []
    for (w,x) in zip(lst, scores):
        # inv = inverse([i])
        s = (w*x)/sum_val
        S.append(s)

    # dataset = np.array(S)
    # print(sum(S))
    return sum(S)

def hybrid(Sa, num_labels):
    A = [0]*num_labels
    for i in range(num_labels):
        s = []
        for j in range(len(Sa)):
            s.append(Sa[j][i])
        # print("S: ", s)
        A[i] = get_zcore_weighted_avg(s)
    return A



def dempster_shafer(line1, line2, num_labels):
    # print("line1: ", line1)
    # print("line2: ", line2)
    A = [0]*num_labels
    denominator = 0
    K = 0
    for i in range(28):
        A[i] = (line1[i] * line2[i])
        denominator = 0
        for j in range(num_labels):
            if i != j:
                denominator += (line1[i] * line2[j])
        K += denominator
    
    denominator = K
    for i in range(num_labels):
        A[i] = (A[i]/denominator)
    return A

def uninorm_aggregation(Sa, num_labels):
    e = 0.5
    A = [0]*num_labels
    n = len(Sa)
    coeff1 = pow((1-e), n-1)    # (1-e)^n-1
    coeff2 = pow(e, n-1)        # e^n-1 
    for i in range(num_labels):
        mul1 = 1
        mul2 = 1
        for j in range(len(Sa)):
            mul1 = (mul1 * Sa[j][i])
            mul2 = (mul2 * (1 - Sa[j][i]))

        denominator = (coeff1 * mul1) + (coeff2 * mul2)
        if denominator == 0:
            A[i] = 0
        else:
            A[i] = ((coeff1 * mul1)/(denominator))
    return A

if __name__ == '__main__':
     print("#Agg. Functions")
    #  print(get_class_map())