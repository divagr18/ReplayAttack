import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import os
import random
import time
import joblib
from sklearn.mixture import GaussianMixture
import speech_recognition as sr
from difflib import SequenceMatcher
from sklearn.preprocessing import StandardScaler
SAMPLE_RATE = 16000
RECORD_SECONDS_ENROLL = 5
RECORD_SECONDS_VERIFY = 4
N_MFCC = 20
N_COMPONENTS = 32
ENROLL_COUNT = 5
PHRASE_SIMILARITY_THRESHOLD = 0.8
INCLUDE_DELTA_FEATURES = True
INCLUDE_SPECTRAL_FEATURES = True
INCLUDE_PITCH_FEATURES = True
BASE_DIR = 'voice_auth_data'
ENROLL_DIR = os.path.join(BASE_DIR, 'enrollment_audio')
VERIFY_DIR = os.path.join(BASE_DIR, 'verification_audio')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_EXT = '.model'
VERIFICATION_THRESHOLD = -60
POSSIBLE_PHRASES = ['open the front door please', 'the sky is blue today',
    'my favorite food is pizza', 'I need to charge my phone',
    'turn on the television', 'where is the nearest station',
    'water is important for life', 'computers make work easier',
    'reading a good book is relaxing', 'hello how are you doing']


def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def record_audio(filename, duration, samplerate=SAMPLE_RATE):
    """Records audio from the microphone and saves it to a WAV file."""
    print(f'Recording for {duration} seconds...')
    num_frames = max(1, int(duration * samplerate))
    recording = sd.rec(num_frames, samplerate=samplerate, channels=1, dtype
        ='int16')
    sd.wait()
    wav.write(filename, samplerate, recording)
    print(f'Recording saved to {filename}')
    return filename


def extract_features(audio_path, n_mfcc=N_MFCC, samplerate=SAMPLE_RATE):
    """Extract raw audio features from a given file path including MFCCs and optional delta, spectral, and pitch features.

Args:
    audio_path (str): Path to the input audio file.
    n_mfcc (int, optional): Number of MFCC coefficients to extract. Defaults to N_MFCC.
    samplerate (int, optional): Sampling rate for audio loading. Defaults to SAMPLE_RATE.

Returns:
    np.ndarray or None: Combined feature matrix of shape (n_frames, n_features) containing raw, unscaled features.
                        Returns None if audio is too short, features cannot be aligned, or an error occurs."""
    try:
        y, sr = librosa.load(audio_path, sr=samplerate)
        if len(y) < 2048:
            print(f'Warning: Audio file {audio_path} is too short.')
            return None
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        features_to_combine = [mfccs]
        if INCLUDE_DELTA_FEATURES:
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            features_to_combine.extend([mfcc_delta, mfcc_delta2])
        mfcc_combined = np.vstack(features_to_combine)
        final_features_list = [mfcc_combined.T]
        if INCLUDE_SPECTRAL_FEATURES:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[
                0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            if spectral_centroids.shape[0] == mfcc_combined.shape[1]:
                final_features_list.append(spectral_centroids.reshape(-1, 1))
                final_features_list.append(spectral_rolloff.reshape(-1, 1))
            else:
                print(
                    f'Warning: Frame mismatch between MFCC and spectral features for {audio_path}. Skipping spectral.'
                    )
        if INCLUDE_PITCH_FEATURES:
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.
                note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0[np.isnan(f0)] = 0
            if f0.shape[0] == mfcc_combined.shape[1]:
                final_features_list.append(f0.reshape(-1, 1))
            else:
                print(
                    f'Warning: Frame mismatch between MFCC and pitch features for {audio_path}. Skipping pitch.'
                    )
                target_frames = mfcc_combined.shape[1]
                if f0.shape[0] > target_frames:
                    f0 = f0[:target_frames]
                    final_features_list.append(f0.reshape(-1, 1))
                elif f0.shape[0] < target_frames:
                    padding = target_frames - f0.shape[0]
                    f0 = np.pad(f0, (0, padding), mode='constant',
                        constant_values=0)
                    final_features_list.append(f0.reshape(-1, 1))
        num_frames = final_features_list[0].shape[0]
        aligned_features = [feat for feat in final_features_list if feat.
            shape[0] == num_frames]
        if len(aligned_features) > 0:
            combined_features = np.hstack(aligned_features)
            return combined_features
        else:
            print(f'Error: No features could be aligned for {audio_path}')
            return None
    except Exception as e:
        print(f'Feature extraction error for {audio_path}: {e}')
        import traceback
        traceback.print_exc()
        return None


def recognize_speech_from_file(audio_path):
    """Transcribes speech from an audio file using Google Web Speech API."""
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        try:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            print(f'Speech recognized: {text}')
            return text
        except sr.WaitTimeoutError:
            print('No speech detected within timeout.')
            return None
        except sr.UnknownValueError:
            print('Google Speech Recognition could not understand audio')
            return None
        except sr.RequestError as e:
            print(f'Could not request results from Google; {e}')
            return None
        except Exception as e:
            print(f'Error during speech recognition: {e}')
            return None


def generate_challenges():
    """Generates 3 random phrases and a simple math problem."""
    phrases = random.sample(POSSIBLE_PHRASES, 3)
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)
    math_problem = f'what is {num1} plus {num2}'
    math_answer = num1 + num2
    return phrases, math_problem, math_answer


def enroll_user(username):
    """Enrolls user: extracts features, fits scaler, trains GMM, saves both."""
    user_enroll_dir = os.path.join(ENROLL_DIR, username)
    user_model_file = os.path.join(MODEL_DIR, f'{username}{MODEL_EXT}')
    if os.path.exists(user_model_file):
        print(f"User '{username}' already enrolled.")
        return
    ensure_dir(user_enroll_dir)
    print(f'\n--- Enrolling User: {username} ---')
    enrollment_phrases = ['Please record this sentence for enrollment.',
        'This is another sample of my voice.',
        'Reading helps capture different sounds.',
        'The quick brown fox jumps over the lazy dog.',
        'Voice authentication systems need good data.']
    print(f'You will need to record your voice {ENROLL_COUNT} times.')
    print('For each recording, please clearly say the requested phrase.')
    all_features = []
    for i in range(ENROLL_COUNT):
        phrase_to_say = enrollment_phrases[i % len(enrollment_phrases)]
        print(
            f"\nRecording {i + 1}/{ENROLL_COUNT}: Please say -> '{phrase_to_say}'"
            )
        input(
            f'Press Enter to start recording sample {i + 1}/{ENROLL_COUNT}...')
        duration = max(RECORD_SECONDS_ENROLL, len(phrase_to_say.split()) * 
            0.5 + 1.5)
        filename = os.path.join(user_enroll_dir, f'enroll_{i + 1}.wav')
        record_audio(filename, duration)
        features = extract_features(filename)
        if features is not None and features.shape[0] > 0:
            all_features.append(features)
        else:
            print(
                f'Failed to extract valid features for recording {i + 1}. Skipping.'
                )
    if len(all_features) < max(1, ENROLL_COUNT // 2):
        print(
            f'Enrollment failed: Only {len(all_features)} valid recordings processed. Need more.'
            )
        return
    try:
        features_matrix = np.vstack(all_features)
        if features_matrix.shape[0] < N_COMPONENTS:
            print(
                f'Warning: Low number of feature frames ({features_matrix.shape[0]}). Model might be unstable.'
                )
        print(
            f'Total feature frames collected: {features_matrix.shape[0]}, Feature dimension: {features_matrix.shape[1]}'
            )
    except ValueError as e:
        print(f'Error stacking features: {e}. Not enough data?')
        return
    scaler = StandardScaler()
    try:
        print('Fitting scaler...')
        scaler.fit(features_matrix)
        print('Transforming features with scaler...')
        scaled_features_matrix = scaler.transform(features_matrix)
    except Exception as e:
        print(f'Error during scaling: {e}')
        return
    print('Training voice model (GMM)...')
    gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type='diag',
        random_state=42, reg_covar=1e-06, n_init=3)
    try:
        gmm.fit(scaled_features_matrix)
        print('GMM training complete.')
    except Exception as e:
        print(f'Error during GMM training: {e}')
        return
    ensure_dir(MODEL_DIR)
    model_data = {'gmm': gmm, 'scaler': scaler}
    try:
        joblib.dump(model_data, user_model_file)
        print(
            f'Enrollment successful! Model and scaler saved for {username} at {user_model_file}'
            )
    except Exception as e:
        print(f'Error saving the model/scaler: {e}')


def verify_user(username):
    """Verifies user: Loads model+scaler, extracts features, scales, scores."""
    user_model_file = os.path.join(MODEL_DIR, f'{username}{MODEL_EXT}')
    user_verify_dir = os.path.join(VERIFY_DIR, username)
    if not os.path.exists(user_model_file):
        print(
            f"Error: Model file for '{username}' not found at {user_model_file}. Please enroll first."
            )
        return False
    try:
        model_data = joblib.load(user_model_file)
        gmm = model_data['gmm']
        scaler = model_data['scaler']
        print('Loaded GMM model and scaler.')
    except Exception as e:
        print(f'Error loading model/scaler from {user_model_file}: {e}')
        return False
    ensure_dir(user_verify_dir)
    print(f'\n--- Verifying User: {username} ---')
    phrases, math_problem, expected_math_answer = generate_challenges()
    print('Please respond to the following prompts:')
    recorded_files = []
    for i, phrase in enumerate(phrases):
        print(f"\nPrompt {i + 1}: Please say -> '{phrase}'")
        input('Press Enter to start recording...')
        duration = max(RECORD_SECONDS_VERIFY, len(phrase.split()) * 0.5 + 1.5)
        filename = os.path.join(user_verify_dir, f'verify_phrase_{i + 1}.wav')
        record_audio(filename, duration=duration)
        recorded_files.append(filename)
        time.sleep(0.2)
    print(f"\nPrompt 4: Please solve and say the answer -> '{math_problem}'")
    input('Press Enter to start recording...')
    filename_math = os.path.join(user_verify_dir, 'verify_math.wav')
    record_audio(filename_math, RECORD_SECONDS_VERIFY + 1)
    recorded_files.append(filename_math)
    time.sleep(0.2)
    print('\n--- Analyzing Responses ---')
    correct_responses = True
    spoken_phrases_text = []
    for i, filename in enumerate(recorded_files[:3]):
        print(f'Recognizing phrase {i + 1}...')
        text = recognize_speech_from_file(filename)
        spoken_phrases_text.append(text)
        if not text:
            print(f'  No speech recognized for phrase {i + 1}.')
            correct_responses = False
            continue
        expected_phrase_lower = phrases[i].replace("'", '').lower()
        recognized_text_lower = text.lower()
        matcher = SequenceMatcher(None, expected_phrase_lower,
            recognized_text_lower)
        similarity = matcher.ratio()
        print(
            f"  Phrase {i + 1}: Expected='{phrases[i]}', Got='{text}', Similarity={similarity:.2f}"
            )
        if similarity < PHRASE_SIMILARITY_THRESHOLD:
            print(
                f'  Mismatch detected for phrase {i + 1} (Similarity below {PHRASE_SIMILARITY_THRESHOLD * 100:.0f}%).'
                )
            correct_responses = False
        else:
            print(f'  Phrase {i + 1} recognized sufficiently.')
    print('Recognizing math answer...')
    spoken_math_text = recognize_speech_from_file(filename_math)
    spoken_math_answer = None
    if spoken_math_text:
        try:
            words_to_digits = {'zero': 0, 'one': 1, 'two': 2, 'three': 3,
                'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
                'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 
                16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
                'twenty': 20}
            cleaned_text = spoken_math_text.lower().strip().replace('-', '')
            if cleaned_text.isdigit():
                spoken_math_answer = int(cleaned_text)
            elif cleaned_text in words_to_digits:
                spoken_math_answer = words_to_digits[cleaned_text]
            else:
                parts = cleaned_text.split()
                if len(parts) == 2 and parts[0] == 'twenty' and parts[1
                    ] in words_to_digits and words_to_digits[parts[1]] < 10:
                    spoken_math_answer = 20 + words_to_digits[parts[1]]
                else:
                    print(
                        f"Could not parse '{spoken_math_text}' as a simple number or 'twenty X'."
                        )
        except ValueError:
            print(
                f"Could not convert recognized answer '{spoken_math_text}' to a number."
                )
    if (spoken_math_answer is None or spoken_math_answer !=
        expected_math_answer):
        print(
            f"  Math answer incorrect or not recognized. Expected: {expected_math_answer}, Got: {spoken_math_answer} (from text: '{spoken_math_text}')"
            )
        correct_responses = False
    else:
        print(f'  Math answer ({spoken_math_answer}) recognized correctly.')
    if not correct_responses:
        print('\nChallenge-response failed. Authentication denied.')
        return False
    print(
        '\nChallenge-response successful. Proceeding to voice verification...')
    verification_features_list = []
    for filename in recorded_files:
        features = extract_features(filename)
        if features is not None and features.shape[0] > 0:
            verification_features_list.append(features)
        else:
            print(f'Warning: Could not extract valid features from {filename}.'
                )
    if not verification_features_list:
        print(
            'Could not extract any valid features from verification recordings. Auth failed.'
            )
        return False
    try:
        verification_features_matrix = np.vstack(verification_features_list)
        print(
            f'Verification features collected: {verification_features_matrix.shape}'
            )
        if verification_features_matrix.shape[0] == 0:
            print('Verification feature matrix is empty. Auth failed.')
            return False
    except ValueError as e:
        print(f'Error stacking verification features: {e}. Auth failed.')
        return False
    try:
        print('Scaling verification features...')
        scaled_verification_features = scaler.transform(
            verification_features_matrix)
        print(
            f'Scaled verification features shape: {scaled_verification_features.shape}'
            )
    except Exception as e:
        print(f'Error scaling verification features: {e}')
        print(
            f'Expected features: {scaler.n_features_in_}, Got: {verification_features_matrix.shape[1]}'
            )
        return False
    try:
        if gmm.n_features_in_ != scaled_verification_features.shape[1]:
            print(
                f'CRITICAL Feature mismatch after scaling: Model trained with {gmm.n_features_in_}, but got {scaled_verification_features.shape[1]} features for verification.'
                )
            return False
        print('Scoring features with GMM...')
        score = gmm.score(scaled_verification_features)
        print(f'Voice match score (log-likelihood): {score:.2f}')
    except Exception as e:
        print(f'Error scoring features with GMM: {e}')
        return False
    if score >= VERIFICATION_THRESHOLD:
        print(f'\nVoice match score >= threshold ({VERIFICATION_THRESHOLD}).')
        print('--- Authentication GRANTED ---')
        return True
    else:
        print(f'\nVoice match score < threshold ({VERIFICATION_THRESHOLD}).')
        print('--- Authentication DENIED ---')
        return False


if __name__ == '__main__':
    ensure_dir(BASE_DIR)
    ensure_dir(ENROLL_DIR)
    ensure_dir(VERIFY_DIR)
    ensure_dir(MODEL_DIR)
    while True:
        print('\n===== Voice Authentication System =====')
        print('1. Enroll New User')
        print('2. Verify Existing User')
        print('3. Exit')
        choice = input('Choose an option (1/2/3): ')
        if choice == '1':
            username = input('Enter username for enrollment: ').strip().lower()
            if username:
                old_model_path = os.path.join(MODEL_DIR,
                    f'{username}{MODEL_EXT}')
                if os.path.exists(old_model_path):
                    print(
                        f'Existing model file found at {old_model_path}. You might need to delete it manually or confirm re-enrollment.'
                        )
                enroll_user(username)
            else:
                print('Username cannot be empty.')
        elif choice == '2':
            username = input('Enter username to verify: ').strip().lower()
            if username:
                verify_user(username)
            else:
                print('Username cannot be empty.')
        elif choice == '3':
            print('Exiting...')
            break
        else:
            print('Invalid choice. Please try again.')
