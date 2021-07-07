import csv
import pprint
import random
import shutil
import signal
import smtplib
import ssl
import sys
import uuid
from email.mime.text import MIMEText
from pathlib import Path
from typing import Tuple, List
from glob import glob
from CNN import CNN
import numpy

import config
from EpochStateData import EpochStateData
from config import *

import pandas as pd
import numpy as np

import pickle

SHOULD_STOP = False


def set_seed(value):
    random.seed(value)
    numpy.random.seed(value)

    if USE_GPU:
        import cupy
        cupy.random.seed(value)


# SET THE SEED TO THE SEED FROM CONFIG NOW
set_seed(SEED)


def result_classifications_to_np_layers(results_classifications: List[int]) -> np.array:
    results = numpy.zeros((len(results_classifications), 10))
    for i in range(len(results_classifications)):
        if not str(results_classifications[i]).isdigit():
            # This is probably a test set. Ignore expected results column
            results = []
            break

        results[i][results_classifications[i] - 1] = 1

    return results


def csv_to_data(path, count=-1) -> Tuple[np.array, np.array]:
    df = pd.read_csv(path, header=None)
    output = df.loc[:, 0]
    data = df.drop(columns=0).to_numpy()
    results_indexes = output.to_numpy()
    results = result_classifications_to_np_layers(results_indexes)
    data = list(data.reshape(data.shape[0], 3, 32, 32))

    if count == -1:
        return data, results
    else:
        return data[:count], results[:count]


def save_state(path: Path, prefix, state: EpochStateData):
    return
    if USE_GPU:
        print("Run was with GPU. Converting state back to numpy before saving")
        weights = [cupy.asnumpy(w) for w in state.weights]
        state = EpochStateData(state.validate_accuracy, state.train_accuracy, state.epoch, weights)

    with open(path / f"{prefix}epoch={state.epoch}_train{state.train_accuracy}%_validate{state.validate_accuracy}% .model", 'wb') as f:
        pickle.dump(state, f)


def load_state(path: Path, net: CNN):
    raise NotImplementedError()
    pickle_model_file = glob(f"{path}/*.model")
    if len(pickle_model_file) != 1:
        raise Exception("Expected only one pickle model file to be found")
    pickle_model_file = pickle_model_file[0]
    with open(pickle_model_file, 'rb') as f:
        state: EpochStateData = pickle.load(f)
        print(f"Loaded state: {state}")

        if USE_GPU:
            print("Run should be with GPU. Converting state to cupy before loading")
            weights = [cupy.array(w) for w in state.weights]
            state.weights = weights

        net.set_weights(state.weights)

    seed_file = glob(f"{path}/seed")
    if len(seed_file) != 1:
        raise Exception("Seed file wasn't found")

    with open(seed_file, 'r') as f:
        seed = int(f.read())
        set_seed(seed)


def get_subset(train_data, train_correct, count):
    random_rows_idx = np.random.choice(train_data.shape[0], size=count, replace=False)
    return train_data[np.ix_(random_rows_idx)], train_correct[np.ix_(random_rows_idx)]


def apply_noise(train_data, prob):
    raise NotImplementedError()
    after_noise_data = EpochStateData.deep_copy_list_of_np_arrays(train_data)
    for i in after_noise_data:
        indices = np.random.choice(np.arange(i.size), replace=False,
                                   size=int(i.size * prob))
        i[indices] = 0
    return after_noise_data


def send_mail(mail, message):
    sender_email = "algobiotester@gmail.com"

    msg = MIMEText(message, _charset="UTF-8")

    port = 465  # For SSL
    password = "algobio1!"

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, mail, msg.as_string())


def shuffle(train_data, train_correct, validate_data, validate_correct):
    data = numpy.concatenate((train_data,validate_data))
    correct = numpy.concatenate((train_correct,validate_correct))
    rand_state = numpy.random.get_state()
    numpy.random.shuffle(data)
    numpy.random.set_state(rand_state)
    numpy.random.shuffle(correct)
    train_data, validate_data = numpy.split(data,[len(train_data)])
    train_correct, validate_correct = numpy.split(correct,[len(train_data)])
    return train_data,train_correct,validate_data,validate_correct


def save_predictions(path, prediction_list):
    with open(path, 'w') as f:
        f.writelines([str(p) for p in prediction_list])


def interrupt_handler(sig, frame):
    answer = input("\nAre you sure you want to stop? [y/N]")
    if answer == "y":
        global SHOULD_STOP
        SHOULD_STOP = True
        print("Will stop at the end of the current epoch")


BEST_TEST_RESULT = 0
# TODO: REMOVE BEFORE SUBMITTING
def run_tests(test_data, net, epoch,output_path,current_validate_accuracy, current_train_accuracy):
    global BEST_TEST_RESULT
    print(f"RUNNING EPOCH {epoch} MODEL ON TEST SET")
    prediction_list = []
    for i, data in enumerate(test_data):
        classification = net.classify_sample(data) + 1
        prediction_list.append(classification)

    print("TODO: REMOVE ME")
    import result_compare
    result = result_compare.check_results(prediction_list)
    if result > BEST_TEST_RESULT:
        BEST_TEST_RESULT = result
        print(f"NEW BEST TEST ON EPOCH {epoch} WITH RESULT {result}%")
        save_state(output_path, f"best_test_until_epoch_{epoch}_with_{result}_",EpochStateData(current_validate_accuracy, current_train_accuracy,epoch,net.weights))


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def train_set(net, data_sets: List[Tuple[np.array, np.array]], shuffle=False, mini_batch_size=1):
    if shuffle:
        numpy.random.shuffle(data_sets)

    count = 0

    for sample, expected_results in data_sets:
        count += 1
        if count % 5 == 0:
            print('\r', end='')
            print(f"Training Progress: {count}/{len(data_sets)}", end='')
            sys.stdout.flush()

        net.train_sample(sample, expected_results)

    print("\rFinished training")
    sys.stdout.flush()


def validate_set(net, data_sets: List[Tuple[np.array, np.array]]):
    correct = 0
    total = 0
    certainty = 0

    count = 0
    for index, (sample, expected_result) in enumerate(data_sets):
        count += 1
        if count % 5 == 0:
            print('\r', end='')
            print(f"Classifying Progress: {count}/{len(data_sets)}", end='')
            sys.stdout.flush()

        result, i = net.validate_sample(sample, expected_result)
        certainty += i
        if result:
            correct += 1
        total += 1

    print("\rFinished classifying")

    average_certainty = float(certainty / total)
    print(f"Average Certainty: {average_certainty}")
    correction = float(correct / total) * 100.0
    print(f"Correct: {correction}%")
    return correction, average_certainty



def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        return

    train_csv = sys.argv[1]
    validate_csv = sys.argv[2]
    test_csv = sys.argv[3] if len(sys.argv) >= 4 else None
    current_train_accuracy=0
    epoch = 0

    print(" ======== Config ==========")
    pprint.pprint(list([(k, v) for (k, v) in config.__dict__.items() if k.isupper()]))
    print(" ==========================")

    shape = ((3, 32, 32), (16, 32, 32))
    net = CNN(shape, config.FULLY_CONNECTED_FEATURE_MAP_DIM, LEARNING_RATE, RANDRANGE)
    csv_results = [["epoch", "LR", "train_accuracy", "train_certainty", "validate_accuracy", "validate_certainty"]]

    #    if SEPARATE_VALIDATE:
    #       validate_data_array, validate_correct_array = separate_data(validate_data,validate_correct)

    output_path = Path(str(uuid.uuid4()) if not TRAINED_NET_DIR else TRAINED_NET_DIR)

    if not TRAINED_NET_DIR:
        print(f"Will write output to {output_path}")

    if TRAINED_NET_DIR and Path(TRAINED_NET_DIR).exists():
        print(f"Taking best values from {TRAINED_NET_DIR}. Pickle mode = {SAVED_MODEL_PICKLE_MODE}")
        load_state(TRAINED_NET_DIR, net)

    if test_csv:
        print("Test csv provided")
        test_data, _ = csv_to_data(test_csv)

    if SHOULD_TRAIN:
        output_path.mkdir(exist_ok=True)
        shutil.copy2("config.py", output_path)
        open(output_path / "seed", "w").write(str(SEED))

        validate_data, validate_correct = csv_to_data(validate_csv)

        signal.signal(signal.SIGINT, interrupt_handler)

        print(f"Reading training data from: {train_csv}")

        train_data, train_correct = csv_to_data(train_csv)

        if SHOULD_SHUFFLE:
            train_data,train_correct,validate_data,validate_correct = shuffle(train_data,train_correct,validate_data,validate_correct)

        print("Starting training...")

        current_validate_accuracy = 0
        overall_best_state = EpochStateData(0, 0, 0, net.weights)

        for epoch in range(EPOCH_COUNT):
            if SHOULD_STOP:
                print("Training interrupt requested. Stopping")
                break

            if ADAPTIVE_LEARNING_RATE_MODE == AdaptiveLearningRateMode.FORMULA:
                net.lr = ADAPTIVE_LEARNING_RATE_FORMULA(epoch)
            elif ADAPTIVE_LEARNING_RATE_MODE == AdaptiveLearningRateMode.PREDEFINED_DICT:
                net.lr = ADAPTIVE_LEARNING_RATE_DICT.get(epoch, net.lr)
            else:
                raise NotImplementedError("Unknown adaptive learning rate mode")

            print(f"Epoch {epoch}")
            print(f"Current LR: {net.lr}")

            if (TAKE_BEST_FROM_VALIDATE or TAKE_BEST_FROM_TRAIN) and (overall_best_state.validate_accuracy > 45):
                print("Take best from:", overall_best_state)
                net.weights = EpochStateData.deep_copy_list_of_np_arrays(overall_best_state.weights)


            if SUBSET_SIZE > 0:
                subset_train, subset_correct = get_subset(train_data, train_correct, SUBSET_SIZE)
                if INPUT_LAYER_NOISE_PROB > 0:
                    print(f"Applying noise of {INPUT_LAYER_NOISE_PROB * 100}% on all inputs")
                    subset_train = apply_noise(subset_train, INPUT_LAYER_NOISE_PROB)
                train_set(net, list(zip(subset_train, subset_correct)), shuffle=True, mini_batch_size=MINI_BATCH_SIZE)

            else:
                if INPUT_LAYER_NOISE_PROB > 0:
                    print(f"Applying noise of {INPUT_LAYER_NOISE_PROB * 100}% on all inputs")
                    after_noise_train = apply_noise(train_data, INPUT_LAYER_NOISE_PROB)
                    train_set(net, list(zip(after_noise_train, train_correct)), shuffle=True, mini_batch_size=MINI_BATCH_SIZE)
                else:
                    train_set(net, list(zip(train_data, train_correct)), shuffle=True, mini_batch_size=MINI_BATCH_SIZE)


            print("======= Train Accuracy =======")
            current_train_accuracy, train_certainty = validate_set(net, list(zip(train_data, train_correct)))

            print("======= Validate Accuracy =======")
            current_validate_accuracy, validate_certainty = validate_set(net, list(zip(validate_data, validate_correct)))

            # TODO: REMOVE ME BEFORE SUBMITTING
            if test_csv:
                run_tests(test_data, net, epoch,output_path,current_validate_accuracy, current_train_accuracy)

            csv_results.append([epoch, net.lr, current_train_accuracy, train_certainty, current_validate_accuracy, validate_certainty])

            if TAKE_BEST_FROM_TRAIN and TAKE_BEST_FROM_VALIDATE:
                if current_validate_accuracy + current_train_accuracy > overall_best_state.train_accuracy + overall_best_state.validate_accuracy:
                    #if current_validate_accuracy >= overall_best_state.validate_accuracy and current_train_accuracy + 2.0 > overall_best_state.train_accuracy:
                    overall_best_state = EpochStateData(current_validate_accuracy, current_train_accuracy, epoch, net.weights)
            elif TAKE_BEST_FROM_TRAIN:
                if current_train_accuracy > overall_best_state.train_accuracy:
                    overall_best_state = EpochStateData(current_validate_accuracy, current_train_accuracy, epoch, net.weights)
            else:
                if current_validate_accuracy > overall_best_state.validate_accuracy:
                    overall_best_state = EpochStateData(current_validate_accuracy, current_train_accuracy, epoch, net.weights)
            if epoch % 50 == 0:
                save_state(output_path, f"epoch_{epoch}", EpochStateData(current_validate_accuracy, current_train_accuracy, epoch, net.weights))
                save_state(output_path, f"best_state_until_epoch_{epoch}", overall_best_state)

        print("Done!")
        print("Saving results, weights...")

        with open(output_path / "results.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_results)

        save_state(output_path, "latest_state", EpochStateData(current_validate_accuracy, current_train_accuracy, epoch, net.weights))
        save_state(output_path, "best_state", overall_best_state)

        #mail_content = f"Finished!\nbest state:\n {overall_best_state}\n CONFIG:\n{open('config.py', 'r').read()}"
        #send_mail("yanirbuznah@gmail.com", mail_content)
        #send_mail("ron.evenm@gmail.com", mail_content)

    if test_csv:
        print("Test csv provided. Classifying...")
        test_data, _ = csv_to_data(test_csv)

        prediction_list = []
        for i, data in enumerate(test_data):
            classification = net.classify_sample(data) + 1
            prediction_list.append(classification)

        print("Saving predicted latest_test.txt")
        save_predictions("latest_test.txt", prediction_list)

        print(prediction_list)
        print(output_path)
        print("TODO: REMOVE ME")
        print("Testing results...")
        import result_compare
        result_compare.check_results(prediction_list)


        prediction_list = []
        net.set_weights(overall_best_state.weights)
        for i, data in enumerate(test_data):
            classification = net.classify_sample(data) + 1
            prediction_list.append(classification)

        print("Saving predicted best_test.txt")

        save_predictions("best_test.txt", prediction_list)

        print(prediction_list)
        print(output_path)
        print("TODO: REMOVE ME")
        print("Testing results...")
        import result_compare
        result_compare.check_results(prediction_list)


if __name__ == '__main__':
    main()
