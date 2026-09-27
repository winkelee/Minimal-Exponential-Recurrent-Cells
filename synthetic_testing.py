import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
import random



def get_batch_ex1(args):

    batch_size = args['batch_size']
    vocab = args['vocab']
    copy_size = args.get("copy_size", 5)
    buffer_size = args.get("buffer_size", 50)
    device = args.get("device", "cuda")

    copy_tensor = torch.randint(0, 10, (batch_size, copy_size), device=device)
    zero_tensor = torch.full((batch_size, buffer_size), vocab["<STABLENOISE>"], device=device)
    marker_tensor = torch.full((batch_size, 1), vocab["<STARTTASK>"], device=device)

    content_tensor = torch.cat((copy_tensor, zero_tensor), dim=1)
    ready_tensor = torch.cat((content_tensor, marker_tensor), dim=1)

    return F.one_hot(ready_tensor, num_classes=len(vocab)).float(), copy_tensor

def get_batch_ex2(args):

    batch_size = args['batch_size']
    vocab = args['vocab']
    copy_size = args.get("copy_size", 5)
    buffer_size = args.get("buffer_size", 50)
    device = args.get("device", "cuda")

    first_noise_ix = vocab["NOISE_0"]
    last_noise_ix = vocab["NOISE_99"]

    # Random noise
    noise_tensor = torch.randint(first_noise_ix, last_noise_ix + 1, (batch_size, buffer_size), device=device )

    # Random digits, sorted so they are increasing
    copy_tensor = torch.randint(0, 10, (batch_size, copy_size), device=device).sort(dim=-1).values

    # Choose positions for the digits.
    # Each row gets its own random positions.
    positions = (torch.rand(batch_size, buffer_size, device=device).argsort(dim=-1)[..., :copy_size].sort(dim=-1).values)

    # Put the digits into the noise
    ready_tensor = noise_tensor.clone()

    ready_tensor.scatter_(dim=-1, index=positions, src=copy_tensor)

    marker_tensor = torch.full(
        (batch_size, 1),
        vocab["<STARTTASK>"],
        device=device,
    )

    ready_tensor = torch.cat([ready_tensor, marker_tensor], dim=-1)

    return F.one_hot(ready_tensor, num_classes=len(vocab)).float(), copy_tensor

def get_batch_ex3(args):

    batch_size = args['batch_size']
    vocab = args['vocab']
    copy_size = args.get("copy_size", 5)
    buffer_size = args.get("buffer_size", 50)
    device = args.get("device", "cuda")

    first_noise_ix = vocab["NOISE_0"]
    last_noise_ix = vocab["NOISE_99"]

    # Random noise
    noise_tensor = torch.randint(first_noise_ix, last_noise_ix + 1, (batch_size, buffer_size), device=device )

    # Random digits
    copy_tensor = torch.randint(0, 10, (batch_size, copy_size), device=device)

    # Choose positions for the digits.
    # Each row gets its own random positions.
    positions = (torch.rand(batch_size, buffer_size, device=device).argsort(dim=-1)[..., :copy_size].sort(dim=-1).values)

    # Put the digits into the noise
    ready_tensor = noise_tensor.clone()

    ready_tensor.scatter_(dim=-1, index=positions, src=copy_tensor)

    marker_tensor = torch.full(
        (batch_size, 1),
        vocab["<STARTTASK>"],
        device=device,
    )

    ready_tensor = torch.cat([ready_tensor, marker_tensor], dim=-1)

    return F.one_hot(ready_tensor, num_classes=len(vocab)).float(), copy_tensor.sort(dim=-1).values


def get_batch_ex4(args):
    #Note - buffer size and copy size are IDENTICAL in this task.
    # buffer_size = 500 means that the model has to READ and WRITE 500 tokens.


    batch_size = args['batch_size']
    vocab = args['vocab']
    buffer_size = args.get("buffer_size", 50)
    bracket_values = args['bracket_values']
    device = args.get("device", "cuda")

    brackets_tensor = bracket_values[torch.randint(len(bracket_values), size=(batch_size, buffer_size), device=device)]
    copy_tensor = torch.flip(brackets_tensor, dims=[1])
    copy_tensor = copy_tensor + 1 # We exploit a cool vocab trick - the closed brackets are just the next index.
    

    marker_tensor = torch.full(
        (batch_size, 1),
        vocab["<STARTTASK>"],
        device=device,
    )

    ready_tensor = torch.cat([brackets_tensor, marker_tensor], dim=-1)

    return F.one_hot(ready_tensor, num_classes=len(vocab)).float(), copy_tensor


def get_batch_ex5(args):


    batch_size = args['batch_size']
    vocab = args['vocab']
    lookup_size = args.get("lookup_size", 5)
    pairs = args.get("pairs", 25)
    device = args.get("device", "cuda")

    first_number_ix = vocab["0"]
    last_number_ix = vocab["9"]
    first_letter_ix = vocab["A"]
    last_letter_ix = vocab["Z"]

    strings = []
    target_strings = []

    for i in range(batch_size):
        string = []
        tracking_set = dict()
        for j in range(pairs):
            letter = random.randint(first_letter_ix, last_letter_ix)
            number = random.randint(first_number_ix, last_number_ix)
            string.append(letter)
            string.append(number)
            string.append(vocab["|"])
            tracking_set[letter] = number
        lookup_indeces = random.choices(list(tracking_set.keys()), k=lookup_size)
        string.append(vocab["<ENDSEQ>"])
        string.extend(lookup_indeces)
        #string.append(vocab["<STARTTASK>"])

        target_string = [tracking_set[letter_ix] for letter_ix in lookup_indeces]

        strings.append(string)
        target_strings.append(target_string)

    ready_tensor = torch.tensor(strings, device=device)
    copy_tensor = torch.tensor(target_strings, device=device)

    return F.one_hot(ready_tensor, num_classes=len(vocab)).float(), copy_tensor

def evaluate(model, copy_size, buffer_size, total_batches, batch_size, vocab, batching_function, batching_args, teacher_forcing=True, diagnostic=False):

    print("Testing model at ", buffer_size, " buffer_size.")

    avg_difference_sum = 0

    for batch_num in trange(total_batches + 1): #We want to get to the end of the user-specified range.

        batch, targets = batching_function(batching_args) #targets = (batch, seq)
        model_targets = F.one_hot(targets, num_classes=len(vocab)).float()

        if teacher_forcing==False:
            logits = model.predict_inference(
                x=batch,
                vocab=vocab,
                max_len=copy_size-1
            )
        else:
                logits = model.forward(batch, model_targets) # (batch, seq, emb)
        tokens = torch.argmax(logits, dim=2) # (batch, seq)

        difference = targets - tokens #we'll get some number where the numbers are not the same

        difference = (difference !=0).float()

        num_elements = copy_size * batch_size

        num_different = torch.sum(difference)

        avg_difference = num_different / num_elements

        avg_difference_sum = avg_difference_sum + avg_difference.item()

        if diagnostic==True and batch_num == total_batches:
            print("True targets: ", targets)
            print("Predictions: ", tokens)

    return avg_difference_sum / (total_batches+1)



def plot_loss_curves(steps, train_history, test_history, description="Loss Curves", model_name="Model"):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_history, label='Train Loss', color='teal', linewidth=2)
    plt.plot(steps, test_history, label='Test Loss', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.title(f'{model_name} {description}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def run_training(model, vocab, total_batches, test_batches, lower_buffer_size, buffer_size, opt, test_break, model_name, model_save_note,
                     steps_history, train_history, test_history, batching_function, batching_args):
    training_loss = 0

    for batch_num in trange(total_batches + 1):

        act_buffer_size = random.randint(lower_buffer_size, buffer_size)

        batch, targets = batching_function(batching_args)

        model_targets = F.one_hot(targets, num_classes=len(vocab)).float()

        #print(model_targets.shape)

        opt.zero_grad()

        logits = model.forward(batch, model_targets)
        loss_t = F.cross_entropy(logits.transpose(1, 2), targets)

        loss_t.backward()

        opt.step()

        training_loss = training_loss + loss_t.item()

        if batch_num % 1000 == 0 and batch_num != 0:
            save_model(model, opt, epoch_saved=batch_num, model_name=model_name, note=model_save_note)

        if batch_num % test_break == 0:

            with torch.no_grad():
                total_loss = 0
                avg_training_loss = training_loss / test_break
                training_loss = 0

                model.eval()

                for test_batch_num in trange(test_batches):
                    batch, targets = batching_function(batching_args)
                    model_targets = F.one_hot(targets, num_classes=len(vocab)).float()
                    logits = model.forward(batch, model_targets)
                    loss_t = F.cross_entropy(logits.transpose(1,2), targets)
                    total_loss = total_loss + loss_t.item()

                model.train()

                avg_testing_loss = total_loss / test_batches

                steps_history.append(batch_num)
                train_history.append(avg_training_loss)
                test_history.append(avg_testing_loss)

                plot_loss_curves(steps_history, train_history, test_history, model_name=model_name)

                print(f"STEP: {batch_num} | TRAIN LOSS: {avg_training_loss:.4f} | TEST LOSS: {avg_testing_loss:.4f}")



def save_model(model, optimizer, epoch_saved, model_name, note):
    checkpoint = {
        "epoch_saved": epoch_saved,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    name = '_'.join(model_name.split()) + "_" + '_'.join(note.split()) + "_epoch_" + str(epoch_saved) + ".pt"
    torch.save(checkpoint, name)

def load_model(filename, device, model, optimizer):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch_saved"] + 1
    return start_epoch