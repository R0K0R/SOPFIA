import numpy as np
import sounddevice as sd
import threading
import reedsolo
import math
import itertools
import time

THRESHOLD_DB = 4  # dB threshold to consider signal present (Can be made configurable)
WINDOW_DURATION = 0.05  # seconds per analysis window (Can be made configurable)
# Removed FREQ_MAP, TARGET_FREQS, END_FREQ - will be dynamic
# FREQ_MAP = { ... }
# TARGET_FREQS = [19000, 19500, 20000]  # frequencies to monitor
# END_FREQ = 18500  # frequency indicating end of transmission
FREQ_TOLERANCE = 100  # Hz tolerance (Can be made configurable)
MIN_SYMBOL_DURATION = 0.08  # seconds to consider a signal stable (Can be made configurable)

bit_sequence = ''
lock = threading.Lock()

def build_freq_map(base, target_freqs):
    """Builds the frequency map based on the provided base and target frequencies.
       Assumes the same assignment logic as test_output.py's generate_freqs_and_map.
    """
    num_freqs = len(target_freqs)
    expected_num_freqs = math.ceil(math.log2(base + 1))
    if num_freqs < expected_num_freqs:
        raise ValueError(f"Provided {num_freqs} target frequencies, but base {base} requires at least {expected_num_freqs}")

    freq_map = {}
    combination_tuples = []
    indices = list(range(num_freqs))
    for i in range(1, num_freqs + 1):
        for combo_indices in itertools.combinations(indices, i):
            combo_tuple = tuple(1 if idx in combo_indices else 0 for idx in indices)
            combination_tuples.append(combo_tuple)

    if len(combination_tuples) < base:
        raise ValueError(f"Could not generate enough unique frequency combinations ({len(combination_tuples)}) for base {base} using {num_freqs} frequencies.")

    digits = [np.base_repr(d, base=base) for d in range(base)]

    for i in range(base):
        freq_map[combination_tuples[i]] = digits[i] # Map tuple -> digit for receiver

    # Add reverse mapping (digit -> tuple) for convenience if needed elsewhere, though receiver primarily uses tuple->digit
    # freq_map_reverse = {v: k for k, v in freq_map.items()}

    return freq_map

def generate_target_freqs(base, start_freq, last_freq):
    """Generates the list of target frequencies based on base, start, and last frequency (inclusive)."""
    if base < 2:
        raise ValueError("Base must be 2 or greater")
    num_freqs = math.ceil(math.log2(base + 1))
    if num_freqs == 1:
        target_freqs = [start_freq]
    else:
        freq_interval = (last_freq - start_freq) / (num_freqs - 1)
        target_freqs = [start_freq + i * freq_interval for i in range(num_freqs)]
    return target_freqs, num_freqs

def record_audio_continuous(target_freqs, freq_map, end_freq, freq_tolerance, threshold_db, 
                            min_symbol_duration, sample_rate=44100, device=None):
    global bit_sequence
    global interval_list
    prev_raw_symbol = None
    stable_duration = 0.0
    last_added_symbol = None
    num_target_freqs = len(target_freqs)
    received_time = time.time()
    interval_list = list() # List of intervals between received symbols

    print("Listening for incoming signal...")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', device=device) as stream:
        while True:
            audio_data, _ = stream.read(int(sample_rate * WINDOW_DURATION))
            audio_data = audio_data.flatten()

            fft_result = np.fft.rfft(audio_data)
            magnitude = np.abs(fft_result)
            magnitude_db = 20 * np.log10(magnitude + 1e-12)
            freqs = np.fft.rfftfreq(len(audio_data), 1 / sample_rate)

            # Dynamically check target frequencies
            mags = []
            for target_f in target_freqs:
                mag = np.max(magnitude_db[(freqs > target_f - freq_tolerance) & (freqs < target_f + freq_tolerance)])
                mags.append(mag)
            
            # Check end frequency
            mag_end = np.max(magnitude_db[(freqs > end_freq - freq_tolerance) & (freqs < end_freq + freq_tolerance)])

            if mag_end > threshold_db:
                print("\nEnd signal detected. Stopping reception.")
                break

            # Create frequency flags tuple based on dynamic check
            freq_flags = tuple(int(m > threshold_db) for m in mags)
            # Use the provided freq_map (tuple -> digit)
            raw_symbol = freq_map.get(freq_flags, None)

            if raw_symbol is None:
                stable_duration = 0.0
                prev_raw_symbol = None
                last_added_symbol = None # Reset last added symbol on silence/invalid signal
                continue

            if raw_symbol == prev_raw_symbol:
                stable_duration += WINDOW_DURATION
            else:
                stable_duration = WINDOW_DURATION  # start counting from first detection

            if stable_duration >= min_symbol_duration:
                with lock:
                    # Only add if the symbol is different from the last added one
                    if raw_symbol != last_added_symbol:
                        bit_sequence += raw_symbol
                        print(raw_symbol, end='', flush=True)
                        interval_list.append(time.time() - received_time) # Add interval to list
                        received_time = time.time()
                        last_added_symbol = raw_symbol # Update last added symbol
                # Reset stability check regardless of whether it was added,
                # to prevent immediate re-triggering on the same continuous tone.
                stable_duration = 0.0
                prev_raw_symbol = None # Reset prev_raw_symbol to force stability re-check
            else:
                prev_raw_symbol = raw_symbol
    
    # Find the error by seeking inappropriate intervals

    interval_list.pop(0)

    for i in range(len(interval_list)):
        interval = interval_list[i]
        average_interval = (sum(interval_list) - interval) / (len(interval_list) - 1)
        if interval > average_interval * 1.75:
            for j in range(round(interval/average_interval - 0.2) - 1):
                print("Inappropriate interval detected: ", interval, "at position", i)
                print("Adding 0 to the subjected bit")
                bit_sequence = bit_sequence[0:(i+1)] + '0' + bit_sequence[(i+1):]

def _try_decode(sequence, base, padding, rs_nsym):
    """Attempts to decode a single base-N sequence. Returns decoded string or None."""
    # Basic check moved here, ensure sequence is not empty and length is correct multiple
    if not sequence or len(sequence) % padding != 0 or len(sequence) == 0:
        return None

    try:
        # Step 1: Convert base-N string to bytes
        byte_values = []
        for i in range(0, len(sequence), padding):
            chunk = sequence[i:i+padding]
            byte_val = int(chunk, base) # Can raise ValueError
            if byte_val < 0 or byte_val > 255:
                 return None # Invalid byte value
            byte_values.append(byte_val)
        encoded_bytes = bytes(byte_values)
        # Check if encoded_bytes is empty, which might happen if byte_values ends up empty
        if not encoded_bytes:
            return None

    except ValueError:
        return None # Error during base conversion
    except Exception: # Catch broader errors during conversion
        return None

    try:
        # Step 2: Reed-Solomon decode
        rs = reedsolo.RSCodec(rs_nsym)
        # Ensure enough data for Reed-Solomon decode (at least nsym + 1 for meaningful result?)
        # RSCodec itself might handle short inputs, but worth noting.
        # Let's assume RSCodec handles it or errors appropriately.
        decoded_data = rs.decode(encoded_bytes) # Can raise ReedSolomonError
        decoded_bytes = decoded_data[0]
    except reedsolo.ReedSolomonError:
        return None # RS decoding error
    except ValueError as e: # RSCodec can raise ValueError too, e.g., nsym out of range
        # print(f"[DEBUG] RSCodec ValueError: {e}")
        return None
    except Exception: # Catch broader errors during RS decoding
        return None

    try:
        # Step 3: Convert bytes to string
        # Check if decoded_bytes is empty before decoding
        if not decoded_bytes:
            # This might indicate only ECC symbols remained, or original data was empty
            return "" # Return empty string for successfully decoded empty data
        return decoded_bytes.decode('utf-8') # Can raise UnicodeDecodeError
    except UnicodeDecodeError:
        return None
    except Exception: # Catch broader errors during final decoding
        return None

def bits_to_text(baseN_sequence, base, padding, rs_nsym):
    """Decodes base-N string to text, trying multiple insertions/deletions if length is wrong."""
    successful_decodings = {} # Use dict to store {decoded_text: source_sequence} and avoid duplicates
    sequences_to_try = {baseN_sequence} # Start with the original sequence
    original_length = len(baseN_sequence)
    processed_sequences = set() # Keep track of sequences already decoded to avoid re-processing

    if original_length == 0:
        print("[INFO] Received empty sequence.")
        return []

    # Check if original length is already correct
    is_length_correct = (original_length % padding == 0)

    if not is_length_correct:
        print(f"[WARNING] Sequence length {original_length} is not a multiple of padding {padding}.")
        print(f"[INFO] Generating candidate sequences via multiple deletions/insertions. This may take a LONG time!")

        # --- Deletion ---
        num_to_delete = original_length % padding
        target_len_del = original_length - num_to_delete
        if num_to_delete > 0 and target_len_del >= 0: # Need something to delete from and target non-negative
             # Limit the number of deletions to avoid extreme computation?
             max_deletions = 4 # Example limit - adjust as needed, or remove for true "everything"
             if num_to_delete <= max_deletions:
                 print(f"[INFO] Trying {num_to_delete} deletion(s) to reach target length {target_len_del}...")
                 count = 0
                 # Generate combinations of indices to remove
                 indices_to_remove_combinations = itertools.combinations(range(original_length), num_to_delete)
                 for indices_to_remove in indices_to_remove_combinations:
                      # Build the string by keeping characters *not* at the specified indices
                      indices_to_remove_set = set(indices_to_remove)
                      deleted_seq = "".join(baseN_sequence[i] for i in range(original_length) if i not in indices_to_remove_set)
                      if len(deleted_seq) == target_len_del: # Double check length
                           sequences_to_try.add(deleted_seq)
                      count += 1
                      if count % 10000 == 0: # Progress indicator for large counts
                           print(f"[DEBUG] Generated {count} deletion candidates...")
                 print(f"[INFO] Generated {count} deletion candidate sequences.")
             else:
                 print(f"[WARNING] Required deletions ({num_to_delete}) exceeds limit ({max_deletions}). Skipping deletion phase.")


        # --- Insertion ---
        num_to_insert = padding - (original_length % padding) if (original_length % padding != 0) else 0
        target_len_ins = original_length + num_to_insert
        if num_to_insert > 0:
            # Limit the number of insertions?
            max_insertions = 4 # Example limit
            if num_to_insert <= max_insertions:
                print(f"[INFO] Trying {num_to_insert} insertion(s) of '0' to reach target length {target_len_ins}...")
                count = 0
                # Insert a block of '0's at every possible position
                insertion_block = '0' * num_to_insert
                for i in range(original_length + 1):
                    inserted_seq = baseN_sequence[:i] + insertion_block + baseN_sequence[i:]
                    if len(inserted_seq) == target_len_ins: # Double check length
                         sequences_to_try.add(inserted_seq)
                    count +=1
                # Note: This only tries inserting a single block of '0's.
                # A truly exhaustive search would insert '0's individually at all combinations of positions,
                # or try inserting *any* base-N digit, which is computationally far more expensive.
                print(f"[INFO] Generated {count} insertion candidate sequences (single block '0' insertion).")
            else:
                 print(f"[WARNING] Required insertions ({num_to_insert}) exceeds limit ({max_insertions}). Skipping insertion phase.")

    # --- Attempt Decoding ---
    print(f"\n[INFO] Attempting decoding for up to {len(sequences_to_try)} candidate sequence(s)...")
    processed_count = 0
    for seq in sequences_to_try:
        # Skip if already processed or if length is incorrect *after* generation (shouldn't happen with current logic, but safe check)
        if seq in processed_sequences or len(seq) % padding != 0 or len(seq) == 0:
             continue

        processed_sequences.add(seq)
        processed_count += 1
        if processed_count % 1000 == 0:
             print(f"[DEBUG] Processed {processed_count} candidates...")

        decoded = _try_decode(seq, base, padding, rs_nsym)
        if decoded is not None:
            # Check if this decoded text is already found from a different sequence
            # This might indicate ambiguity, which could be interesting to know
            existing_source = successful_decodings.get(decoded)
            if existing_source is None:
                 print(f"[SUCCESS] Decoded '{decoded}' from sequence (len={len(seq)}): '{seq[:30]}{'...' if len(seq)>30 else ''}'")
                 successful_decodings[decoded] = seq # Store first sequence that produced this text
            elif existing_source != seq:
                 # Optional: Log ambiguity
                 print(f"[WARNING] Ambiguity detected: Text '{decoded}' can be decoded from multiple sequences:")
                 print(f"          - Original: (len={len(existing_source)}) '{existing_source[:30]}...'")
                 print(f"          - Current:  (len={len(seq)}) '{seq[:30]}...'")
                 # Decide how to handle ambiguity - keep first? Store all sources? For now, keep first.

    print(f"[INFO] Finished processing {processed_count} unique candidate sequences.")
    return list(successful_decodings.keys())

def select_input_device():
    print("Available input devices:")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"ID {idx}: {device['name']} (Default SR: {device['default_samplerate']} Hz)")
    selected = input("Enter input device ID (leave blank for default): ")
    return 1 if selected.strip() == '' else int(selected)

def main():
    global bit_sequence
    device = select_input_device()
    sample_rate = 44100

    print("\n--- Receiver Configuration ---")
    print("Please enter parameters matching the transmitter:")
    base = input("Base used by transmitter (e.g., 7): ")
    start_freq = input("Start Frequency used by transmitter (Hz): ")
    last_freq = input("Last Frequency used by transmitter (Hz): ")
    end_freq = input("End Signal Frequency (Hz): ")
    # padding = int(input(f"Base-{base} to Byte Padding length used by transmitter: "))
    rs_nsym = input("Number of Reed-Solomon symbols used by transmitter: ")

    base = 31 if base == '' else int(base)
    start_freq = 18000 if start_freq == '' else float(start_freq)
    last_freq = 20000 if last_freq == '' else float(last_freq)
    end_freq = 17500 if end_freq == '' else float(end_freq)
    rs_nsym = 10 if rs_nsym == '' else int(rs_nsym)

    freq_tolerance = FREQ_TOLERANCE
    threshold_db = THRESHOLD_DB
    min_symbol_duration = MIN_SYMBOL_DURATION

    # Automatically calculate padding
    padding = math.ceil(math.log(256, base))
    print(f"[INFO] Using padding={padding} for base-{base} (auto-calculated)")

    try:
        target_freqs, num_freqs = generate_target_freqs(base, start_freq, last_freq)
        print(f"Generated {num_freqs} target frequencies: {target_freqs}")
        freq_map = build_freq_map(base, target_freqs)
        print(f"Built frequency map (combination: digit): {freq_map}")
    except ValueError as e:
        print(f"Error processing configuration: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during configuration: {e}")
        return

    listen_thread = threading.Thread(
        target=record_audio_continuous, 
        kwargs={
            'sample_rate': sample_rate, 
            'device': device,
            'target_freqs': target_freqs,
            'freq_map': freq_map,
            'end_freq': end_freq,
            'freq_tolerance': freq_tolerance,
            'threshold_db': threshold_db,
            'min_symbol_duration': min_symbol_duration
        }
    )
    listen_thread.start()
    listen_thread.join()

    print(f"Listener stopped.") # Added listener stopped message
    print(f"Final raw sequence received (len={len(bit_sequence)}): '{bit_sequence}'")
    print(f"Interval list: {interval_list}")

    if not bit_sequence:
         print("No sequence was detected.")
    else:
         possible_texts = bits_to_text(bit_sequence, base, padding, rs_nsym)

         if possible_texts:
             print(f"\n--- Successfully Decoded Text(s) ---")
             for i, text in enumerate(possible_texts):
                 print(f"Result {i+1}: {text}")
             print("------------------------------------")
         else:
             # Recalculate here if the original length was correct
             original_length_was_correct = (len(bit_sequence) % padding == 0)
             if original_length_was_correct:
                  print("\nCould not decode the received sequence (length was correct, but RS decode or UTF-8 failed).")
             else:
                  print("\nCould not successfully decode the received sequence, even with corrections.")

if __name__ == "__main__":
    main()
