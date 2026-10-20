# Nepali Currency Classifier for Blind Users

By Kushal (GitHub: kushal4510), IOE Pashchimanchal Campus, Pokhara

**Live demo:** https://huggingface.co/spaces/kushal4510/nepali-currency-classifier

A phone-friendly web app that looks at a Nepali currency note and **speaks the amount out loud**. It is built for blind and low-vision users, and it runs fully inside the browser. No server and no app install are needed.

## What it does

1. You open the page on your phone.
2. The app asks for **3 photos** of the same note, one by one, and speaks each prompt.
3. The model looks at each photo and makes a guess.
4. The app takes a **majority vote** of the 3 guesses.
5. It speaks the answer, for example: "This is 500 rupees". If the photos do not agree, it asks you to try again.

The model knows 7 notes: Rs 5, 10, 20, 50, 100, 500 and 1000.

## Why 3 photos every time

My first idea was simple: if the model is very sure about one photo, answer straight away, and ask for more photos only when it is unsure. While testing, I found a problem with this idea. A folded 50 rupee note was predicted as "10 rupees" with **100% confidence**. A confident wrong answer never triggers the "ask for more photos" rule.

So I changed the design: the app **always** asks for exactly 3 photos and needs more than half of them to agree. A confident-wrong single photo now gets outvoted by the other two. This matters most for a blind user, because a wrong answer about money is worse than asking for one more photo.

## Accessibility

- Works with **TalkBack (Android)** and **VoiceOver (iPhone)**, the screen readers that blind users use to navigate their phones.
- One big, clearly labeled button, so it is easy to find by touch and by screen reader.
- All instructions and results are **spoken aloud** using the browser's built-in speech.
- Colors adapt to dark mode.

## Tech stack

| Part | Tool |
| --- | --- |
| Model training | PyTorch in Google Colab |
| Final model | ResNet18 (transfer learning) |
| Baseline model | 3-block CNN built from scratch |
| Dataset | Kaggle Nepali currency dataset (7 classes, about 19,600 training images) plus my own phone photos |
| Model export | ONNX |
| In-browser inference | ONNX Runtime Web (JavaScript) |
| Voice output | Browser SpeechSynthesis (web app), gTTS (Colab notebooks) |
| Hosting | Hugging Face Static Space (free) |

## Two models, compared honestly

I built two models on purpose, so I could see what transfer learning really changes.

**Model 1: CNN from scratch.** 3 convolution blocks (3 to 16 to 32 to 64 channels), each with BatchNorm, ReLU and MaxPool, then Dropout and a linear layer. Input is 128x128. It learned everything from the Kaggle data only.

**Model 2: ResNet18 with transfer learning.** Pretrained on ImageNet. I replaced the last layer with a 7-class layer and fine-tuned it. Input is 224x224 with ImageNet normalization.

### Results

|  | CNN from scratch | ResNet18 |
| --- | --- | --- |
| Kaggle test accuracy | about 95.6% | about 99% |
| My real phone photos | **42.6%** | **96%** (see note below) |

The from-scratch CNN looked fine on clean dataset images, but it **fell apart on real phone photos**.

### CNN on my real phone photos, per note

| Note | Accuracy |
| --- | --- |
| Rs 5 | 12% |
| Rs 10 | 85% |
| Rs 20 | 15% |
| Rs 50 | 58% |
| Rs 100 | 67% |
| Rs 500 | 14% |
| Rs 1000 | 8% |

My guess is that the CNN learned shallow clues from the dataset (like overall color or size) instead of real design details, so it breaks when lighting, angle and background change.

### A fairness note

The CNN **never saw** my phone photos, so its 42.6% is a fully fair test. ResNet18 was trained with **most** of my photos mixed in, so its 96% is not a purely held-out number. I report it anyway because it shows the improvement the real-photo training gave, but please read it with that in mind. A cleaner test would use brand new photos that neither model has seen.

## The journey: how the ResNet got to 96%

1. **Clean dataset, easy win.** The model scored very high on Kaggle test images and on clean screenshots.
2. **Real phone photos failed.** On real camera photos it failed again and again. This is a domain gap: dataset photos and phone photos look different.
3. **Data augmentation** (random crop, rotation, perspective, color jitter) helped a little but did not fix it.
4. **Unfreezing layer4** (the most task-specific ResNet layer) with a gentle learning rate also was not enough alone.
5. **My own photos.** I took about 150 to 160 real photos, sorted by denomination, and mixed them into training. Because they were few compared to Kaggle, I repeated them about 15 times. Real-photo accuracy went from about 72% to about 79%, then kept climbing with more epochs.
6. **Targeted fix for a weak spot.** Testing showed that **folded notes in dark light** (100 and 1000) were confidently predicted as 10 rupees. I took 5 to 8 extra photos per note in exactly that situation and retrained. Accuracy went up to 96%.
7. **Voting system.** Added the always-3-photos majority vote (see above).
8. **Deployment.** Hugging Face now needs a paid plan for Gradio and Docker Spaces, so I converted the model to ONNX and built a plain HTML and JavaScript page that runs the model inside the browser. This is free, fast, and private, because photos never leave the phone.

### Training details

- ResNet18 backbone mostly frozen. Only **layer4** and the new final layer were trained.
- **Differential learning rates:** 1e-4 for layer4 (already pretrained, needs gentle changes) and 1e-3 for the new final layer (random start, needs faster learning).
- BatchNorm layers stayed frozen, because my dataset is too small to update their statistics safely.
- Cross-entropy loss for 7 classes.
- Images cached once instead of being reopened and resized every epoch.
- Checkpoints saved to Google Drive so training could resume after Colab disconnects.

## Repository contents

- **CNN from scratch notebook:** training of the baseline model.
- **ResNet18 training notebook:** transfer learning, own-photo mixing, augmentation, checkpoints.
- **Testing notebooks:** load a saved model, take 3 photos, vote, and speak the result. There is one for ResNet18 and one for the from-scratch CNN, plus an optional cell that measures accuracy on a folder of real photos.
- **.gitignore:** keeps credentials (like kaggle.json), caches and large model files out of the repo.

The model files (.pth and .onnx) are **not** stored here because they are large. The ONNX model and the web page (index.html) live in the Hugging Face Space linked at the top.

## How to try it

**Easiest:** open the live demo on your phone and follow the spoken instructions.

**With the notebooks:**

1. Open a testing notebook in Google Colab and choose a CPU runtime.
2. Upload the matching model file when asked.
3. Run the cells from top to bottom and add 3 photos of one note.

## Limitations

- The model only knows the 7 current note types in my dataset, not coins or old notes.
- Very dark, blurry or heavily crumpled notes can still fail. The voting system reduces this but cannot remove it.
- Real-photo accuracy was measured on my own photos, which is a small set. More testers and more photos would give a stronger number.
- Voice output is in English for now.

## What I learned

- A high accuracy on a clean dataset does not mean the model works in the real world. Testing on real phone photos exposed problems that the dataset score hid.
- Transfer learning is a huge help when data is limited. The same task went from 42.6% to 96% on real photos.
- A confidence score can be confidently wrong, so a safety design (voting) matters as much as the model.
- Targeted data collection beats collecting more of the same data.
- Building for blind users means testing with screen readers and speaking everything, not only making a model work.

## Future work

- Voice prompts and results in **Nepali**.
- Test with real blind and low-vision users and improve from their feedback.
- Collect more varied photos (more lighting, more folds, old notes) and measure on a clean held-out set.
- Try a lighter model such as MobileNet for faster loading on slow phones.

## Author

**Kushal**, Bachelor's in Electronics, Communication and Information Engineering, IOE Pashchimanchal Campus. GitHub: kushal4510
