# SVG Diagrams for WaveTutor educational dashboard
import textwrap


def get_longitudinal_wave_diagram() -> str:
    return textwrap.dedent("""
    <svg viewBox="0 0 500 150" width="100%" height="150" style="background: transparent; color: inherit;">
        <!-- Compressions & Rarefactions (Lines representing particles) -->
        <g stroke="currentColor" stroke-width="1.5" opacity="0.8">
            <!-- Compression 1 (0 to 60) -->
            <line x1="10" y1="20" x2="10" y2="100" />
            <line x1="25" y1="20" x2="25" y2="100" />
            <line x1="35" y1="20" x2="35" y2="100" />
            <line x1="40" y1="20" x2="40" y2="100" />
            <line x1="43" y1="20" x2="43" y2="100" />
            <line x1="45" y1="20" x2="45" y2="100" />
            <line x1="47" y1="20" x2="47" y2="100" />
            <line x1="50" y1="20" x2="50" y2="100" />
            <line x1="53" y1="20" x2="53" y2="100" />
            <line x1="55" y1="20" x2="55" y2="100" />
            <line x1="57" y1="20" x2="57" y2="100" />
            <line x1="60" y1="20" x2="60" y2="100" />
            <line x1="65" y1="20" x2="65" y2="100" />
            <line x1="75" y1="20" x2="75" y2="100" />
            <line x1="90" y1="20" x2="90" y2="100" />

            <!-- Rarefaction 1 (90 to 170) -->
            <line x1="110" y1="20" x2="110" y2="100" />
            <line x1="135" y1="20" x2="135" y2="100" />
            <line x1="160" y1="20" x2="160" y2="100" />
            <line x1="185" y1="20" x2="185" y2="100" />

            <!-- Compression 2 (185 to 260) -->
            <line x1="205" y1="20" x2="205" y2="100" />
            <line x1="220" y1="20" x2="220" y2="100" />
            <line x1="230" y1="20" x2="230" y2="100" />
            <line x1="235" y1="20" x2="235" y2="100" />
            <line x1="238" y1="20" x2="238" y2="100" />
            <line x1="240" y1="20" x2="240" y2="100" />
            <line x1="242" y1="20" x2="242" y2="100" />
            <line x1="245" y1="20" x2="245" y2="100" />
            <line x1="248" y1="20" x2="248" y2="100" />
            <line x1="250" y1="20" x2="250" y2="100" />
            <line x1="252" y1="20" x2="252" y2="100" />
            <line x1="255" y1="20" x2="255" y2="100" />
            <line x1="260" y1="20" x2="260" y2="100" />
            <line x1="270" y1="20" x2="270" y2="100" />
            <line x1="285" y1="20" x2="285" y2="100" />

            <!-- Rarefaction 2 (285 to 365) -->
            <line x1="305" y1="20" x2="305" y2="100" />
            <line x1="330" y1="20" x2="330" y2="100" />
            <line x1="355" y1="20" x2="355" y2="100" />
            <line x1="380" y1="20" x2="380" y2="100" />

            <!-- Compression 3 (380 to 450) -->
            <line x1="400" y1="20" x2="400" y2="100" />
            <line x1="415" y1="20" x2="415" y2="100" />
            <line x1="425" y1="20" x2="425" y2="100" />
            <line x1="430" y1="20" x2="430" y2="100" />
            <line x1="435" y1="20" x2="435" y2="100" />
            <line x1="440" y1="20" x2="440" y2="100" />
            <line x1="445" y1="20" x2="445" y2="100" />
            <line x1="450" y1="20" x2="450" y2="100" />
            <line x1="465" y1="20" x2="465" y2="100" />
            <line x1="480" y1="20" x2="480" y2="100" />
        </g>

        <!-- Wave Direction Indicator -->
        <path d="M 10 135 H 350" stroke="#3b82f6" stroke-width="2" stroke-dasharray="4 4" fill="none" />
        <polygon points="350,131 360,135 350,139" fill="#3b82f6" />
        <text x="12" y="128" fill="#3b82f6" font-size="10" font-family="sans-serif" font-weight="bold">Direction of Wave Travel</text>

        <!-- Particle Motion Indicator (Centered Arrow & Text) -->
        <g stroke="#ef4444" stroke-width="2">
            <line x1="392" y1="135" x2="488" y2="135" />
            <polygon points="397,132 389,135 397,138" fill="#ef4444" />
            <polygon points="483,132 491,135 483,138" fill="#ef4444" />
        </g>
        <text x="440" y="128" text-anchor="middle" fill="#ef4444" font-size="9" font-family="sans-serif" font-weight="bold">Particle Motion</text>

        <!-- Dynamic Labels -->
        <text x="40" y="15" fill="currentColor" font-size="10" font-family="sans-serif" font-weight="bold">Compression</text>
        <text x="125" y="15" fill="currentColor" font-size="10" font-family="sans-serif" font-weight="bold">Rarefaction</text>
        <text x="235" y="15" fill="currentColor" font-size="10" font-family="sans-serif" font-weight="bold">Compression</text>
        <text x="320" y="15" fill="currentColor" font-size="10" font-family="sans-serif" font-weight="bold">Rarefaction</text>
    </svg>
    """)


def get_shear_wave_diagram() -> str:
    return textwrap.dedent("""
    <svg viewBox="0 0 500 150" width="100%" height="150" style="background: transparent; color: inherit;">
        <!-- Sine Wave Line -->
        <path d="M 10 75 Q 70 15, 130 75 T 250 75 T 370 75 T 490 75" fill="none" stroke="currentColor" stroke-width="3" />

        <!-- Wave Direction Indicator -->
        <path d="M 10 135 H 320" stroke="#3b82f6" stroke-width="2" stroke-dasharray="4 4" fill="none" />
        <polygon points="320,131 330,135 320,139" fill="#3b82f6" />
        <text x="12" y="128" fill="#3b82f6" font-size="10" font-family="sans-serif" font-weight="bold">Direction of Wave Travel</text>

        <!-- Particle Motion Indicator on the right (Solid Vertical Double Arrow) -->
        <g stroke="#ef4444" stroke-width="2">
            <line x1="365" y1="110" x2="365" y2="140" />
            <polygon points="362,113 365,105 368,113" fill="#ef4444" />
            <polygon points="362,137 365,145 368,137" fill="#ef4444" />
        </g>
        <text x="375" y="129" fill="#ef4444" font-size="9" font-family="sans-serif" font-weight="bold">Particle Motion</text>

        <!-- Neutral line -->
        <line x1="10" y1="75" x2="490" y2="75" stroke="currentColor" stroke-width="1" stroke-dasharray="10 5" opacity="0.3" />
    </svg>
    """)


def get_through_transmission_diagram() -> str:
    return textwrap.dedent("""
    <svg viewBox="0 0 500 150" width="100%" height="150" style="background: transparent; color: inherit;">
        <!-- Specimen Body -->
        <rect x="150" y="25" width="200" height="90" fill="currentColor" fill-opacity="0.08" stroke="currentColor" stroke-width="2" />

        <!-- Specimen label outside of the box pointing to the box -->
        <text x="250" y="14" text-anchor="middle" fill="currentColor" font-size="11" font-family="sans-serif" font-weight="bold">Specimen</text>
        <g stroke="currentColor" stroke-width="1">
            <line x1="250" y1="16" x2="250" y2="22" />
            <polygon points="248,20 250,23 252,20" fill="currentColor" />
        </g>

        <!-- Transmitter (Left) -->
        <rect x="90" y="40" width="60" height="60" rx="3" fill="#3b82f6" fill-opacity="0.9" stroke="currentColor" stroke-width="1.5" />
        <text x="96" y="70" fill="white" font-size="10" font-family="sans-serif" font-weight="bold">Transmitter</text>
        <text x="115" y="85" fill="white" font-size="12" font-family="sans-serif" font-weight="bold">(T)</text>

        <!-- Receiver (Right) -->
        <rect x="350" y="40" width="60" height="60" rx="3" fill="#10b981" fill-opacity="0.9" stroke="currentColor" stroke-width="1.5" />
        <text x="362" y="70" fill="white" font-size="10" font-family="sans-serif" font-weight="bold">Receiver</text>
        <text x="375" y="85" fill="white" font-size="12" font-family="sans-serif" font-weight="bold">(R)</text>

        <!-- Wave Path Arrow moving through sample -->
        <path d="M 160 70 H 340" stroke="#f59e0b" stroke-width="3" stroke-dasharray="5 3" fill="none" />
        <polygon points="335,65 345,70 335,75" fill="#f59e0b" />
        <text x="215" y="58" fill="#f59e0b" font-size="10" font-family="sans-serif" font-weight="bold">Wave Path (c)</text>

        <!-- Dimension Line for d (centered label) -->
        <line x1="150" y1="130" x2="350" y2="130" stroke="currentColor" stroke-width="1.5" />
        <line x1="150" y1="125" x2="150" y2="135" stroke="currentColor" stroke-width="1.5" />
        <line x1="350" y1="125" x2="350" y2="135" stroke="currentColor" stroke-width="1.5" />
        <text x="250" y="125" text-anchor="middle" fill="currentColor" font-size="11" font-family="sans-serif" font-weight="bold">Thickness (d)</text>
    </svg>
    """)


def get_manual_peak_selection_diagram() -> str:
    return textwrap.dedent("""
    <svg viewBox="0 0 500 160" width="100%" height="160" style="background: transparent; color: inherit;">
        <!-- Excitation Waveform (Top) -->
        <path d="M 10 40 L 30 40 Q 40 10, 50 40 T 70 40 T 90 40 L 250 40" fill="none" stroke="#3b82f6" stroke-width="2" />
        <!-- Selected Peak Dot (Excitation) -->
        <circle cx="50" cy="10" r="5" fill="#ef4444" />
        <text x="60" y="15" fill="currentColor" font-size="10" font-family="sans-serif">Selected Peak (Excitation)</text>

        <!-- Received Waveform (Bottom) -->
        <path d="M 10 110 L 180 110 Q 190 80, 200 110 T 220 110 T 240 110 L 250 110" fill="none" stroke="#10b981" stroke-width="2" />
        <!-- Selected Peak Dot (Received) -->
        <circle cx="200" cy="80" r="5" fill="#ef4444" />
        <text x="210" y="85" fill="currentColor" font-size="10" font-family="sans-serif">Selected Peak (Receiver)</text>

        <!-- Dimension Line for Time Delay -->
        <path d="M 50 50 V 130" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3" opacity="0.5" />
        <path d="M 200 85 V 130" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3" opacity="0.5" />

        <line x1="50" y1="125" x2="200" y2="125" stroke="#ef4444" stroke-width="1.5" />
        <polygon points="55,122 47,125 55,128" fill="#ef4444" />
        <polygon points="195,122 203,125 195,128" fill="#ef4444" />
        <text x="80" y="120" fill="#ef4444" font-size="10" font-family="sans-serif" font-weight="bold">Time-of-Flight (t)</text>
    </svg>
    """)


def get_signal_alignment_diagram() -> str:
    return textwrap.dedent("""
    <svg viewBox="0 0 500 150" width="100%" height="150" style="background: transparent; color: inherit;">
        <!-- Left Subplot: Raw Signals -->
        <g transform="translate(10, 0)">
            <rect x="0" y="10" width="220" height="110" fill="currentColor" fill-opacity="0.03" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3" />
            <text x="10" y="25" fill="currentColor" font-size="9" font-family="sans-serif" font-weight="bold">Shifted (Raw Signals)</text>

            <!-- Excitation Burst -->
            <path d="M 10 70 L 30 70 Q 40 40, 50 70 T 70 70 T 90 70 L 210 70" fill="none" stroke="#3b82f6" stroke-width="1.5" />
            <!-- Received Burst -->
            <path d="M 10 70 L 110 70 Q 120 50, 130 70 T 150 70 T 170 70 L 210 70" fill="none" stroke="#10b981" stroke-width="1.5" />

            <!-- Delay Arrow -->
            <line x1="50" y1="85" x2="130" y2="85" stroke="#ef4444" stroke-width="1" />
            <polygon points="53,83 47,85 53,87" fill="#ef4444" />
            <polygon points="127,83 133,85 127,87" fill="#ef4444" />
            <text x="75" y="96" fill="#ef4444" font-size="8" font-family="sans-serif" font-weight="bold">Offset</text>
        </g>

        <!-- Right Subplot: Aligned Envelopes -->
        <g transform="translate(260, 0)">
            <rect x="0" y="10" width="220" height="110" fill="currentColor" fill-opacity="0.03" stroke="currentColor" stroke-width="1" stroke-dasharray="3 3" />
            <text x="10" y="25" fill="currentColor" font-size="9" font-family="sans-serif" font-weight="bold">Aligned Envelopes</text>

            <!-- Excitation Envelope (Dashed) -->
            <path d="M 10 70 Q 30 20, 50 20 T 90 70" fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="2 2" />
            <!-- Aligned Received Envelope (Solid) -->
            <path d="M 10 70 Q 30 25, 50 25 T 90 70" fill="none" stroke="#10b981" stroke-width="1.5" />

            <!-- CorelPeak Marker -->
            <circle cx="50" cy="20" r="3" fill="#ef4444" />
            <text x="60" y="25" fill="#ef4444" font-size="8" font-family="sans-serif" font-weight="bold">Aligned</text>
        </g>
    </svg>
    """)
