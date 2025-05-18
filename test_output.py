import numpy as np
import sounddevice as sd
import threading
import reedsolo
import math
import itertools

def int_to_baseN_padded(number, base, padding):
    """Converts an integer to a base-N string with leading zeros."""
    if number == 0:
        return '0' * padding
    # Use np.base_repr directly and then pad
    baseN_str = np.base_repr(number, base=base)
    return baseN_str.zfill(padding)

def generate_tone(frequencies, duration, sample_rate=44100, volume=0.5, fade_time=0.005):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)
    
    for freq in frequencies:
        waveform += np.sin(2 * np.pi * freq * t)

    waveform /= len(frequencies)

    fade_samples = int(sample_rate * fade_time)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    envelope = np.ones_like(waveform)
    envelope[:fade_samples] *= fade_in
    envelope[-fade_samples:] *= fade_out

    waveform *= envelope
    waveform *= volume
    return waveform

def get_key_by_value(d, target_value):
    for key, value in d.items():
        if value == target_value:
            return key
    return None

def generate_freqs_and_map(base, start_freq, end_freq):
    """Generates target frequencies and a mapping for a given base, using start and end frequency."""
    if base < 2:
        raise ValueError("Base must be 2 or greater")
    num_freqs = math.ceil(math.log2(base + 1))
    if num_freqs == 1:
        target_freqs = [start_freq]
    else:
        freq_interval = (end_freq - start_freq) / (num_freqs - 1)
        target_freqs = [start_freq + i * freq_interval for i in range(num_freqs)]

    # Generate frequency map { digit_str: combination_tuple }
    freq_map = {}
    # Generate all combinations of indices (0 to num_freqs-1) representing active frequencies
    # Order combinations: 1 active freq, then 2, then 3... up to num_freqs
    combination_tuples = []
    indices = list(range(num_freqs))
    for i in range(1, num_freqs + 1):
        for combo_indices in itertools.combinations(indices, i):
            combo_tuple = tuple(1 if idx in combo_indices else 0 for idx in indices)
            combination_tuples.append(combo_tuple)

    # Assign combinations to digits '0' through 'base-1'
    if len(combination_tuples) < base:
        raise ValueError(f"Could not generate enough unique frequency combinations ({len(combination_tuples)}) for base {base}")

    digits = [np.base_repr(d, base=base) for d in range(base)] # Get digit representation '0', '1', ..., 'N-1'

    for i in range(base):
        freq_map[digits[i]] = combination_tuples[i]

    return target_freqs, freq_map, num_freqs

def tones_from_baseN(digit, duration, sample_rate, volume, fade_time, target_freqs, freq_map):
    freqs_tuple = freq_map.get(str(digit)) # Ensure digit is string and use the map
    if freqs_tuple is None:
        print(f"Warning: Digit '{digit}' not found in frequency map. Skipping tone generation.")
        return np.zeros(int(sample_rate * duration)) # Return silence if digit is invalid

    active_frequencies = []
    for i in range(len(freqs_tuple)):
        if freqs_tuple[i] == 1:
            active_frequencies.append(target_freqs[i])

    if not active_frequencies:
        # This might happen if the map incorrectly contains an all-zero tuple
        print(f"Warning: No active frequencies for digit '{digit}'. Generating silence.")
        return np.zeros(int(sample_rate * duration))

    return generate_tone(active_frequencies, duration, sample_rate, volume, fade_time)

def text_to_baseN(text, base, nsym=2):
    """Encodes text to a base-N string using Reed-Solomon.

    Returns:
        tuple: (baseN_string, padding_length)
    """
    data_bytes = text.encode('utf-8')
    rs = reedsolo.RSCodec(nsym)
    encoded_bytes = rs.encode(data_bytes)

    # Determine padding length m such that base^m >= 256
    if base == 1:
         raise ValueError("Base cannot be 1")
    padding = math.ceil(math.log(256, base)) # Or ceil(8 / log2(base))

    baseN_list = []
    for byte in encoded_bytes:
        baseN_list.append(int_to_baseN_padded(byte, base, padding))

    return "".join(baseN_list), padding

def build_audio_buffer(baseN_sequence, sample_rate, duration_per_symbol, silence_duration, volume, 
                       target_freqs, freq_map, end_freq, end_tone_duration=0.2, fade_time=0.005):
    silence = np.zeros(int(sample_rate * silence_duration))
    buffer_list = [silence.copy()] # Use a list first

    for digit in baseN_sequence:
        tone = tones_from_baseN(digit, duration_per_symbol, sample_rate, volume, fade_time, target_freqs, freq_map)
        buffer_list.append(tone)
        buffer_list.append(silence.copy())

    # Append end signal
    end_tone = generate_tone([end_freq], end_tone_duration, sample_rate, volume, fade_time)
    buffer_list.append(end_tone)
    buffer_list.append(silence.copy()) # End with silence

    # Concatenate everything into the final buffer
    audio_buffer = np.concatenate(buffer_list)

    # Apply a final fade-out to the very end of the entire buffer
    fade_samples = int(sample_rate * fade_time)
    if len(audio_buffer) >= fade_samples:
        final_fade_out = np.linspace(1, 0, fade_samples)
        audio_buffer[-fade_samples:] *= final_fade_out
    else: 
        # Buffer is shorter than fade time, fade out the whole thing
        final_fade_out = np.linspace(1, 0, len(audio_buffer))
        audio_buffer[:] *= final_fade_out

    return audio_buffer

def play_buffer(audio_buffer, sample_rate=44100, device=None):
    sd.play(audio_buffer, samplerate=sample_rate, device=device)
    sd.wait()

def select_output_device():
    print("Available output devices:")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            print(f"ID {idx}: {device['name']} (Default SR: {device['default_samplerate']} Hz)")
    selected = input("Enter output device ID (leave blank for default): ")
    return None if selected.strip() == '' else int(selected)

def main():
    # Get user inputs
    user_input = input("Enter text to encode: ")
    base = input("Enter desired base (e.g., 7, 10): ")
    start_freq = input("Enter starting frequency (Hz, e.g., 18000): ")
    last_freq = input("Enter last frequency (Hz, e.g., 20000): ")
    end_freq = input("Enter end signal frequency (Hz, e.g., 17000): ")
    duration_per_symbol = input("Enter duration per symbol (s, e.g., 0.1): ")
    silence_duration = input("Enter silence duration between symbols (s, e.g., 0.05): ")
    volume = input("Enter volume (0.0 to 1.0): ")
    rs_nsym = input("Enter number of Reed-Solomon error correction symbols (e.g., 2): ")

    base = 31 if base == '' else int(base)
    start_freq = 18000 if start_freq == '' else float(start_freq)
    last_freq = 20000 if last_freq == '' else float(last_freq)
    end_freq = 17500 if end_freq == '' else float(end_freq)
    duration_per_symbol = 0.1 if duration_per_symbol == '' else float(duration_per_symbol)
    silence_duration = 0.1 if silence_duration == '' else float(silence_duration)
    volume = 1.0 if volume == '' else float(volume)
    rs_nsym = 10 if rs_nsym == '' else int(rs_nsym)
    # Generate frequencies and map
    try:
        target_freqs, freq_map, num_freqs = generate_freqs_and_map(base, start_freq, last_freq)
        print(f"\nGenerated {num_freqs} target frequencies: {target_freqs}")
        print(f"Generated frequency map (digit: combination): {freq_map}")
    except ValueError as e:
        print(f"Error generating frequencies/map: {e}")
        return

    # Encode text
    try:
        baseN_string, padding = text_to_baseN(user_input, base, nsym=rs_nsym)
        print(f"Encoded base-{base} string (padding={padding}): {baseN_string}")
        # Crucial: Inform user about receiver requirements
        print(f"\n--- IMPORTANT --- ")
        print(f"Receiver must be configured for:")
        print(f"  - Base: {base}")
        print(f"  - Frequencies: {target_freqs}")
        print(f"  - End Frequency: {end_freq}")
        print(f"  - Symbol Map: As printed above")
        print(f"  - Base-{base} to Byte Padding: {padding}")
        print(f"  - RS Symbols: {rs_nsym}")
        print(f"-----------------")

    except ValueError as e:
        print(f"Error encoding text: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during encoding: {e}")
        return

    # Select device
    device = select_output_device()
    sample_rate = 44100 # Or get from device info?

    # Build buffer
    print("\nBuilding audio buffer...")
    audio_buffer = build_audio_buffer(
        baseN_sequence=baseN_string, 
        sample_rate=sample_rate, 
        duration_per_symbol=duration_per_symbol, 
        silence_duration=silence_duration, 
        volume=volume, 
        target_freqs=target_freqs, 
        freq_map=freq_map, 
        end_freq=end_freq
    )
    print("Audio buffer built.")

    # Play audio
    print("Playing audio...")
    play_thread = threading.Thread(target=play_buffer, args=(audio_buffer,), kwargs={'sample_rate': sample_rate, 'device': device})
    play_thread.start()
    play_thread.join()
    print("Playback finished.")

if __name__ == "__main__":
    main()
