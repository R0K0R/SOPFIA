# SOPFIA: SOund Protocol For Information Access

SOPFIA (SOund Protocol For Information Access) is a near-field communication protocol designed to transmit data using sound waves via standard speakers and microphones. It utilizes frequencies at the edge of or just beyond the human audible range (e.g., 18kHz - 22kHz), making it unobtrusive while remaining compatible with most commercial audio hardware.

The primary advantage of SOPFIA is its ability to enable data exchange between various devices without requiring specialized hardware modules or complex pairing procedures often associated with technologies like NFC or Bluetooth. It is well-suited for sharing short pieces of textual information, such as URLs, simple messages, or authentication tokens.

## Features

*   **Near-Ultrasonic Frequencies:** Operates in frequency bands like 18kHz-22kHz to minimize audible interference.
*   **Standard Hardware:** Works with commonly available speakers and microphones found in laptops, smartphones, and other devices.
*   **No Complex Pairing:** Eliminates the need for explicit pairing between devices.
*   **Error Correction:** Implements Reed-Solomon codes to enhance data integrity and correct errors that may occur during acoustic transmission.
*   **Adaptive Parameter Prediction (Conceptual):** Designed with a mechanism where the receiver can predict and adapt to certain transmission parameters (e.g., precise frequency start points, symbol duration, data base) based on a set of core pre-defined parameters (6 fixed base frequencies with constant spacing, fixed signal intervals, Reed-Solomon symbol number of 10). This aims to improve robustness in one-to-many communication scenarios, accommodating variations in microphone performance by potentially adjusting bandwidth settings or signal intervals.

## How it Works

SOPFIA operates through a defined process for transmitting and receiving data:

### Transmission Process

1.  **Text Input:** The process begins with the text data to be transmitted.
2.  **UTF-8 Encoding:** The input text is converted into UTF-8 bytes (`text.encode('utf-8')`).
3.  **Reed-Solomon Encoding:** The UTF-8 byte array is encoded using Reed-Solomon codes to add redundancy for error correction (`rs.encode(data_bytes)`).
4.  **Base-N Conversion & Padding:** Each byte of the encoded data is converted into a Base-N representation. Padding is applied as necessary. For example, a byte (256 possible values) might be represented by two Base-32 symbols.
5.  **Frequency Mapping:** The sequence of Base-N symbols is mapped to a sequence of specific frequency tones (or combinations of tones).
6.  **Tone Generation & Transmission:** The mapped frequency tones are generated as sound waves and transmitted via a speaker.

### Reception Process

1.  **Audio Input:** The microphone captures incoming audio signals.
2.  **FFT Analysis & Signal Detection:** The received audio stream is analyzed using Fast Fourier Transform (FFT) to identify the energy levels across different frequencies.
3.  **Symbol Demapping:** Frequency combinations with energy levels exceeding a certain threshold are detected and demapped back into Base-N symbols.
4.  **Base-N Sequence Restoration:** The identified Base-N symbols form a sequence.
5.  **Base-N Decoding & Padding Removal:** The Base-N symbol sequence is decoded back into a byte array, and any padding is removed.
6.  **Reed-Solomon Decoding:** The received byte array undergoes Reed-Solomon decoding to correct potential errors (`rs.decode(received_bytes)`).
7.  **UTF-8 Decoding:** The corrected byte array is decoded from UTF-8 back into the original text (`decoded_bytes.decode('utf-8')`).
8.  **Text Output:** The successfully decoded text is made available.

The protocol's design includes an adaptive mechanism where the receiver, using pre-defined core parameters (6 fixed frequencies, constant inter-frequency spacing, fixed signal intervals, and a Reed-Solomon symbol number of 10), actively predicts and adjusts to other transmission parameters like the exact starting frequency, symbol duration, and data base by analyzing the incoming signal. This feature is intended to enhance reliability in diverse environments and across different devices.

## Development Environment

*   **Programming Language:** Python 3.x
*   **Key Libraries:** NumPy, SoundDevice (and a library for Reed-Solomon code implementation)
*   **Hardware:** Standard built-in or external microphone and speakers.
*   **Audio Sampling Rate:** 48,000 Hz

## Project Status

SOPFIA is currently in an experimental, proof-of-concept stage. Initial tests indicate that stable data transmission is achievable in low-noise environments, and the Reed-Solomon error correction codes are effective in mitigating minor errors. While the transmission speed is currently modest (tens to hundreds of bps), it is potentially sufficient for use cases involving small amounts of data, similar to QR codes but without requiring visual scanning.

## Future Work / Roadmap

The following are potential directions for future research and development:

*   **Enhance Transmission Speed & Stability:**
    *   Explore advanced modulation techniques (e.g., DPSK, multi-level FSK, OFDM adapted for acoustic channels).
    *   Develop adaptive algorithms that dynamically adjust parameters (modulation, symbol length, FEC strength) based on real-time channel conditions.
    *   Investigate multi-tone or frequency hopping spread spectrum (FHSS) techniques for improved robustness and throughput.
*   **Strengthen Error Control:**
    *   Evaluate more powerful error correction codes (e.g., LDPC, Turbo codes).
    *   Consider Hybrid ARQ (Automatic Repeat reQuest) schemes.
*   **Improve User Experience (UX):**
    *   Develop intuitive user interfaces with clear feedback on transmission status.
*   **Explore Diverse Applications:**
    *   Beyond QR code alternatives, investigate uses in offline payments, smart device control, access control systems, and information kiosks.
*   **Enhance Security:**
    *   Implement a lightweight encryption layer for transmitting sensitive information.

## Contributing

Pull requests and contributions are welcome. For major changes, please open an issue first to discuss what you would like to change. Please ensure to update tests as appropriate.

## Author

*   25-095 이호준 (Lee Ho-jun)
    *   Korea Science Academy

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (though a `LICENSE` file itself has not been created yet, the intention is to use MIT). 