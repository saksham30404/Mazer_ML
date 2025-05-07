import numpy as np
import pretty_midi
import soundfile as sf
from typing import Optional, Tuple, List, Dict
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import requests
import re
import random
import argparse
from fastapi.middleware.cors import CORSMiddleware
from scipy.io import wavfile
from scipy.signal import sawtooth, square
from scipy import signal
import time
from tqdm import tqdm

class MusicGenerator:
    def __init__(self):
        self.sample_rate = 44100
        self.note_frequencies = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        
        # Enhanced instrument definitions with richer harmonics and effects
        self.instruments = {
            'piano': {
                'harmonics': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05],
                'attack': 0.002,
                'decay': 0.1,
                'sustain': 0.7,
                'release': 0.3,
                'reverb': 0.4,
                'chorus': 0.2,
                'vibrato': 0.1,
                'stereo_width': 0.3
            },
            'bass': {
                'harmonics': [1.0, 0.6, 0.4, 0.3, 0.2],
                'attack': 0.01,
                'decay': 0.1,
                'sustain': 0.8,
                'release': 0.2,
                'reverb': 0.2,
                'chorus': 0.1,
                'vibrato': 0.05,
                'stereo_width': 0.1
            },
            'saxophone': {
                'harmonics': [1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1],
                'attack': 0.02,
                'decay': 0.1,
                'sustain': 0.8,
                'release': 0.2,
                'reverb': 0.5,
                'chorus': 0.3,
                'vibrato': 0.2,
                'stereo_width': 0.2
            },
            'guitar': {
                'harmonics': [1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1],
                'attack': 0.005,
                'decay': 0.1,
                'sustain': 0.7,
                'release': 0.2,
                'reverb': 0.3,
                'chorus': 0.4,
                'vibrato': 0.15,
                'stereo_width': 0.4
            },
            'synth': {
                'harmonics': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
                'attack': 0.01,
                'decay': 0.1,
                'sustain': 0.8,
                'release': 0.3,
                'reverb': 0.5,
                'chorus': 0.6,
                'vibrato': 0.1,
                'stereo_width': 0.5
            },
            'strings': {
                'harmonics': [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                'attack': 0.1,
                'decay': 0.2,
                'sustain': 0.9,
                'release': 0.4,
                'reverb': 0.6,
                'chorus': 0.3,
                'vibrato': 0.25,
                'stereo_width': 0.6
            },
            'flute': {
                'harmonics': [1.0, 0.9, 0.7, 0.5, 0.4, 0.3],
                'attack': 0.03,
                'decay': 0.1,
                'sustain': 0.8,
                'release': 0.2,
                'reverb': 0.4,
                'chorus': 0.2,
                'vibrato': 0.15,
                'stereo_width': 0.2
            },
            'harp': {
                'harmonics': [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
                'attack': 0.005,
                'decay': 0.2,
                'sustain': 0.6,
                'release': 0.3,
                'reverb': 0.5,
                'chorus': 0.2,
                'vibrato': 0.1,
                'stereo_width': 0.3
            },
            'violin': {
                'harmonics': [1.0, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                'attack': 0.1,
                'decay': 0.2,
                'sustain': 0.9,
                'release': 0.4,
                'reverb': 0.6,
                'chorus': 0.3,
                'vibrato': 0.3,
                'stereo_width': 0.4
            },
            'cello': {
                'harmonics': [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                'attack': 0.1,
                'decay': 0.2,
                'sustain': 0.9,
                'release': 0.4,
                'reverb': 0.6,
                'chorus': 0.3,
                'vibrato': 0.25,
                'stereo_width': 0.3
            },
            'trumpet': {
                'harmonics': [1.0, 0.9, 0.7, 0.5, 0.4, 0.3],
                'attack': 0.03,
                'decay': 0.1,
                'sustain': 0.8,
                'release': 0.2,
                'reverb': 0.4,
                'chorus': 0.2,
                'vibrato': 0.2,
                'stereo_width': 0.2
            },
            'drums': {
                'kick': {'frequency': 60, 'decay': 0.1, 'punch': 0.3, 'body': 0.4},
                'snare': {'frequency': 200, 'decay': 0.2, 'snap': 0.4, 'body': 0.3},
                'hihat': {'frequency': 1000, 'decay': 0.05, 'brightness': 0.6, 'noise': 0.4},
                'tom': {'frequency': 150, 'decay': 0.15, 'resonance': 0.4, 'body': 0.3},
                'clap': {'frequency': 300, 'decay': 0.1, 'spread': 0.5, 'snap': 0.3},
                'ride': {'frequency': 800, 'decay': 0.2, 'brightness': 0.5, 'noise': 0.3}
            }
        }
        
        # Style definitions with unique characteristics
        self.styles = {
            'jazz': {
                'description': 'Smooth jazz with rich harmonies and improvisation',
                'tempo': 90,
                'instruments': ['piano', 'bass', 'drums', 'saxophone'],
                'mix_levels': {
                    'piano': 0.4,
                    'bass': 0.5,
                    'drums': 0.4,
                    'saxophone': 0.6
                },
                'progressions': [
                    ['Dm7', 'G7', 'Cmaj7'],
                    ['Em7', 'A7', 'Dmaj7'],
                    ['Am7', 'D7', 'Gmaj7'],
                    ['Dm7', 'G7', 'Em7', 'A7']
                ],
                'scales': {
                    'mystery': ['C', 'D', 'D#', 'F#', 'G', 'A#'],
                    'blues': ['C', 'D#', 'F', 'F#', 'G', 'A#'],
                    'bebop': ['C', 'D', 'E', 'F', 'G', 'A', 'A#', 'B'],
                    'smooth': ['C', 'D', 'E', 'G', 'A'],
                    'noir': ['C', 'D#', 'F', 'F#', 'G#', 'A#']
                }
            },
            'electronic': {
                'description': 'Modern electronic music with deep bass and atmospheric synths',
                'tempo': 128,
                'instruments': ['synth', 'bass', 'drums', 'electric_piano'],
                'mix_levels': {
                    'synth': 0.7,
                    'bass': 0.6,
                    'drums': 0.5,
                    'electric_piano': 0.3
                },
                'progressions': [
                    ['Cmin', 'G#maj', 'A#maj', 'Fmin'],
                    ['D#maj', 'Cmin', 'G#maj', 'A#maj'],
                    ['Fmin', 'Cmin', 'G#maj', 'D#maj']
                ],
                'scales': {
                    'mystery': ['C', 'D', 'D#', 'F#', 'G', 'A#'],
                    'energy': ['C', 'D', 'E', 'G', 'A'],
                    'dreamy': ['C', 'D', 'F', 'G', 'A#'],
                    'techno': ['C', 'D#', 'F', 'G', 'A#'],
                    'house': ['C', 'D', 'E', 'F', 'G', 'A']
                }
            },
            'rock': {
                'description': 'Energetic rock music with powerful guitar riffs',
                'tempo': 120,
                'instruments': ['guitar', 'bass', 'drums', 'piano'],
                'mix_levels': {
                    'guitar': 0.7,
                    'bass': 0.5,
                    'drums': 0.6,
                    'piano': 0.3
                },
                'progressions': [
                    ['E5', 'B5', 'C#5', 'A5'],
                    ['A5', 'E5', 'F#5', 'D5'],
                    ['G5', 'D5', 'Em5', 'C5']
                ],
                'scales': {
                    'mystery': ['E', 'F#', 'G', 'A', 'B', 'C#'],
                    'energy': ['E', 'F#', 'G#', 'A', 'B', 'C#'],
                    'intensity': ['E', 'F#', 'G', 'A', 'B', 'C'],
                    'metal': ['E', 'F', 'G', 'G#', 'B', 'C'],
                    'punk': ['E', 'G', 'A', 'B', 'D']
                }
            },
            'ambient': {
                'description': 'Atmospheric ambient music with evolving textures',
                'tempo': 60,
                'instruments': ['synth', 'piano', 'strings', 'harp'],
                'mix_levels': {
                    'synth': 0.5,
                    'piano': 0.4,
                    'strings': 0.6,
                    'harp': 0.3
                },
                'progressions': [
                    ['Cmin', 'G#maj', 'A#maj', 'Fmin'],
                    ['D#maj', 'Cmin', 'G#maj', 'A#maj'],
                    ['Fmin', 'Cmin', 'G#maj', 'D#maj']
                ],
                'scales': {
                    'mystery': ['C', 'D', 'D#', 'F#', 'G', 'A#'],
                    'peace': ['C', 'D', 'F', 'G', 'A'],
                    'dreamy': ['C', 'D', 'F', 'G', 'A#'],
                    'space': ['C', 'D#', 'F', 'G#', 'A#'],
                    'meditation': ['C', 'D', 'F', 'G', 'A#']
                }
            },
            'hiphop': {
                'description': 'Urban hip-hop beats with groovy basslines',
                'tempo': 95,
                'instruments': ['drums', 'bass', 'synth', 'piano'],
                'mix_levels': {
                    'drums': 0.7,
                    'bass': 0.6,
                    'synth': 0.4,
                    'piano': 0.3
                },
                'progressions': [
                    ['Cmin', 'G#maj', 'A#maj', 'Fmin'],
                    ['D#maj', 'Cmin', 'G#maj', 'A#maj'],
                    ['Fmin', 'Cmin', 'G#maj', 'D#maj']
                ],
                'scales': {
                    'mystery': ['C', 'D', 'D#', 'F#', 'G', 'A#'],
                    'energy': ['C', 'D', 'E', 'G', 'A'],
                    'groove': ['C', 'D', 'F', 'G', 'A#'],
                    'trap': ['C', 'D#', 'F', 'G', 'A#'],
                    'oldschool': ['C', 'D', 'E', 'G', 'A']
                }
            },
            'classical': {
                'description': 'Elegant classical music with orchestral arrangements',
                'tempo': 80,
                'instruments': ['piano', 'strings', 'harp', 'flute'],
                'mix_levels': {
                    'piano': 0.5,
                    'strings': 0.6,
                    'harp': 0.4,
                    'flute': 0.5
                },
                'progressions': [
                    ['C', 'G', 'Am', 'F'],
                    ['Dm', 'G', 'C', 'Am'],
                    ['F', 'C', 'G', 'Am']
                ],
                'scales': {
                    'mystery': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
                    'elegance': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
                    'drama': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
                    'baroque': ['C', 'D', 'E', 'F', 'G', 'A'],
                    'romantic': ['C', 'D', 'E', 'G', 'A', 'B']
                }
            },
            'lofi': {
                'description': 'Relaxing lo-fi beats with jazzy elements',
                'tempo': 85,
                'instruments': ['piano', 'bass', 'drums', 'synth'],
                'mix_levels': {
                    'piano': 0.5,
                    'bass': 0.4,
                    'drums': 0.3,
                    'synth': 0.4
                },
                'progressions': [
                    ['Cmaj7', 'Am7', 'Dm7', 'G7'],
                    ['Fmaj7', 'Em7', 'Dm7', 'G7'],
                    ['Am7', 'Dm7', 'G7', 'Cmaj7']
                ],
                'scales': {
                    'chill': ['C', 'D', 'E', 'G', 'A'],
                    'study': ['C', 'D', 'F', 'G', 'A#'],
                    'sleep': ['C', 'D#', 'F', 'G', 'A#'],
                    'relax': ['C', 'D', 'E', 'G', 'A'],
                    'jazzy': ['C', 'D', 'E', 'F', 'G', 'A', 'B']
                }
            },
            'synthwave': {
                'description': '80s-inspired retro electronic music',
                'tempo': 110,
                'instruments': ['synth', 'bass', 'drums', 'electric_piano'],
                'mix_levels': {
                    'synth': 0.7,
                    'bass': 0.5,
                    'drums': 0.4,
                    'electric_piano': 0.3
                },
                'progressions': [
                    ['Am', 'F', 'C', 'G'],
                    ['Dm', 'Am', 'F', 'G'],
                    ['C', 'G', 'F', 'Am']
                ],
                'scales': {
                    'retro': ['C', 'D', 'E', 'F', 'G', 'A'],
                    'cyber': ['C', 'D#', 'F', 'G', 'A#'],
                    'neon': ['C', 'D', 'E', 'G', 'A'],
                    'arcade': ['C', 'D', 'F', 'G', 'A#'],
                    'night': ['C', 'D#', 'F', 'G#', 'A#']
                }
            },
            'folk': {
                'description': 'Traditional folk music with acoustic elements',
                'tempo': 100,
                'instruments': ['guitar', 'violin', 'bass', 'flute'],
                'mix_levels': {
                    'guitar': 0.6,
                    'violin': 0.5,
                    'bass': 0.4,
                    'flute': 0.5
                },
                'progressions': [
                    ['C', 'Am', 'F', 'G'],
                    ['G', 'Em', 'C', 'D'],
                    ['Am', 'F', 'C', 'G']
                ],
                'scales': {
                    'celtic': ['C', 'D', 'E', 'G', 'A'],
                    'country': ['C', 'D', 'E', 'F', 'G', 'A'],
                    'bluegrass': ['C', 'D', 'E', 'G', 'A', 'B'],
                    'acoustic': ['C', 'D', 'F', 'G', 'A'],
                    'traditional': ['C', 'D', 'E', 'F', 'G', 'A']
                }
            }
        }
        
        # Enhanced natural language mappings
        self.style_keywords = {
            'jazz': ['jazz', 'jazzy', 'swing', 'blues', 'bebop', 'smooth', 'saxophone', 'improv'],
            'electronic': ['electronic', 'edm', 'techno', 'dance', 'synth', 'electro', 'house', 'trance', 'dubstep'],
            'rock': ['rock', 'metal', 'hard', 'guitar', 'band', 'punk', 'alternative', 'indie', 'grunge'],
            'ambient': ['ambient', 'atmospheric', 'chill', 'relaxing', 'peaceful', 'calm', 'meditation', 'space'],
            'hiphop': ['hiphop', 'hip hop', 'rap', 'trap', 'beats', 'urban', 'rhythm', 'rhyme'],
            'classical': ['classical', 'orchestra', 'orchestral', 'symphony', 'piano', 'elegant', 'baroque', 'romantic'],
            'lofi': ['lofi', 'lo-fi', 'lo fi', 'study', 'beats', 'chill', 'relax', 'sleep'],
            'synthwave': ['synthwave', 'retro', 'retrowave', '80s', 'cyber', 'neon', 'arcade', 'outrun'],
            'folk': ['folk', 'acoustic', 'traditional', 'celtic', 'country', 'bluegrass', 'americana']
        }
        
        self.emotion_keywords = {
            'mystery': ['mystery', 'mysterious', 'dark', 'enigmatic', 'suspense', 'thriller', 'unknown'],
            'energy': ['energy', 'energetic', 'powerful', 'strong', 'upbeat', 'dynamic', 'lively'],
            'dreamy': ['dreamy', 'dream', 'ethereal', 'floating', 'soft', 'gentle', 'airy'],
            'blues': ['blues', 'bluesy', 'soulful', 'melancholic', 'sad', 'emotional', 'deep'],
            'bebop': ['bebop', 'fast', 'complex', 'virtuosic', 'skilled', 'technical', 'intricate'],
            'peace': ['peace', 'peaceful', 'calm', 'serene', 'tranquil', 'quiet', 'still'],
            'groove': ['groove', 'groovy', 'funky', 'rhythmic', 'moving', 'danceable', 'smooth'],
            'intensity': ['intensity', 'intense', 'heavy', 'aggressive', 'fierce', 'strong', 'powerful'],
            'elegance': ['elegance', 'elegant', 'graceful', 'sophisticated', 'refined', 'delicate', 'poised'],
            'drama': ['drama', 'dramatic', 'epic', 'grand', 'powerful', 'theatrical', 'intense'],
            'chill': ['chill', 'relaxed', 'mellow', 'laid back', 'easy', 'smooth', 'cool'],
            'retro': ['retro', 'vintage', 'old school', 'classic', 'nostalgic', 'throwback'],
            'celtic': ['celtic', 'irish', 'scottish', 'folk', 'traditional', 'ancient'],
            'cyber': ['cyber', 'cyberpunk', 'futuristic', 'digital', 'electronic', 'tech'],
            'trap': ['trap', 'modern', 'urban', 'street', 'dark', 'heavy'],
            'space': ['space', 'cosmic', 'stellar', 'galactic', 'floating', 'ethereal'],
            'metal': ['metal', 'heavy', 'aggressive', 'intense', 'powerful', 'dark'],
            'punk': ['punk', 'raw', 'energetic', 'rebellious', 'aggressive', 'fast'],
            'house': ['house', 'dance', 'club', 'groove', 'electronic', 'upbeat'],
            'techno': ['techno', 'electronic', 'dance', 'rhythmic', 'mechanical', 'digital'],
            'noir': ['noir', 'dark', 'mysterious', 'smoky', 'jazzy', 'moody'],
            'smooth': ['smooth', 'mellow', 'cool', 'relaxed', 'gentle', 'easy'],
            'meditation': ['meditation', 'zen', 'peaceful', 'calm', 'mindful', 'tranquil']
        }
        
        self.duration_keywords = {
            'short': 30,
            'medium': 60,
            'long': 90,
            'brief': 20,
            'quick': 15,
            'extended': 120,
            'minute': 60,
            'hour': 3600,
            'sec': 1,
            'half': 30,
            'full': 60
        }

        # Context keywords for better understanding
        self.context_keywords = {
            'time_multipliers': {
                'minute': 60,
                'minutes': 60,
                'hour': 3600,
                'hours': 3600,
                'second': 1,
                'seconds': 1,
                'mins': 60,
                'min': 60,
                'sec': 1,
                'secs': 1,
                'hr': 3600,
                'hrs': 3600
            },
            'quantity_multipliers': {
                'half': 0.5,
                'quarter': 0.25,
                'double': 2,
                'triple': 3,
                'twice': 2,
                'few': 3,
                'couple': 2
            }
        }

    def _create_envelope(self, duration: float, params: Dict) -> np.ndarray:
        """Create ADSR envelope with improved smoothing"""
        samples = int(duration * self.sample_rate)
        attack = int(params['attack'] * self.sample_rate)
        decay = int(params['decay'] * self.sample_rate)
        release = int(params['release'] * self.sample_rate)
        sustain_level = params['sustain']
        
        envelope = np.zeros(samples)
        
        # Attack with exponential curve for smoother onset
        if attack > 0:
            envelope[:attack] = np.exp(np.linspace(-5, 0, attack)) - np.exp(-5)
            envelope[:attack] /= 1 - np.exp(-5)  # Normalize to [0,1]
        
        # Decay with natural curve
        if decay > 0:
            decay_curve = np.exp(np.linspace(0, -2, decay))
            decay_curve = (decay_curve - decay_curve[-1]) / (1 - decay_curve[-1])
            decay_curve = 1 + (sustain_level - 1) * (1 - decay_curve)
            envelope[attack:attack+decay] = decay_curve
        
        # Sustain
        envelope[attack+decay:-release] = sustain_level
        
        # Release with exponential curve for natural fade
        if release > 0:
            release_curve = np.exp(np.linspace(0, -5, release))
            envelope[-release:] = release_curve * sustain_level
        
        # Apply subtle smoothing to avoid clicks
        envelope = signal.savgol_filter(envelope, 15, 3)
        
        return envelope

    def _generate_note(self, frequency: float, duration: float, instrument: Dict) -> np.ndarray:
        """Generate a note with improved audio quality and harmonics"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        note = np.zeros_like(t)
        
        # Generate harmonics with proper ratios and phases
        for i, amplitude in enumerate(instrument['harmonics']):
            # Add slight detuning for richness
            detune = 1 + (random.random() - 0.5) * 0.001
            # Add phase variation for more natural sound
            phase = random.random() * 2 * np.pi
            harmonic = amplitude * np.sin(2 * np.pi * frequency * (i + 1) * detune * t + phase)
            note += harmonic
        
        # Apply ADSR envelope with smoother curves
        envelope = self._create_envelope(duration, instrument)
        note *= envelope
        
        # Add subtle vibrato for more expressive sound
        if 'vibrato' not in instrument or instrument.get('vibrato', 0.0) > 0:
            vibrato_rate = 5.0  # 5 Hz vibrato
            vibrato_depth = 0.005  # Subtle depth
            vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
            note *= vibrato
        
        # Apply anti-aliasing filter
        nyquist = self.sample_rate / 2
        cutoff = min(frequency * 10, nyquist - 100)  # Prevent aliasing
        b, a = signal.butter(4, cutoff / nyquist)
        note = signal.filtfilt(b, a, note)
        
        return note

    def _generate_drum_sound(self, drum_type: str, duration: float) -> np.ndarray:
        """Generate improved drum sounds with better synthesis"""
        params = self.instruments['drums'][drum_type]
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        if drum_type == 'kick':
            # Improved kick with frequency sweep and body
            freq_sweep = params['frequency'] * np.exp(-30 * t)  # Frequency sweep
            sound = np.sin(2 * np.pi * freq_sweep * t)
            # Add body to kick
            body_freq = params['frequency'] * 2
            body = 0.5 * np.sin(2 * np.pi * body_freq * t)
            sound = sound + body
            
        elif drum_type == 'snare':
            # Improved snare with noise and tone
            noise = np.random.uniform(-1, 1, len(t))
            # Filter the noise for more natural sound
            b, a = signal.butter(2, 2000 / (self.sample_rate/2))
            noise = signal.filtfilt(b, a, noise)
            # Add tonal component
            tone = np.sin(2 * np.pi * params['frequency'] * t)
            sound = 0.5 * noise + 0.5 * tone
            
        else:  # hihat
            # Improved hi-hat with filtered noise and resonance
            noise = np.random.uniform(-1, 1, len(t))
            # Multi-band filtering for more realistic sound
            b1, a1 = signal.butter(4, 5000 / (self.sample_rate/2), 'high')
            b2, a2 = signal.butter(4, 12000 / (self.sample_rate/2), 'low')
            noise = signal.filtfilt(b1, a1, noise)
            noise = signal.filtfilt(b2, a2, noise)
            sound = noise
        
        # Apply envelope
        envelope = np.exp(-t / params['decay'])
        # Add quick attack
        attack = int(0.001 * self.sample_rate)
        envelope[:attack] = np.linspace(0, 1, attack)
        sound *= envelope
        
        # Apply subtle saturation for warmth
        sound = np.tanh(sound * 1.5)
        
        return sound

    def _get_chord_notes(self, chord: str) -> List[float]:
        """Convert chord symbol to frequencies with improved voicing"""
        root = chord[0]
        if len(chord) > 1 and chord[1] == '#':
            root = chord[:2]
            quality = chord[2:]
        else:
            quality = chord[1:]
        
        base_freq = self.note_frequencies[root]
        if 'maj7' in quality:
            return [base_freq, base_freq*5/4, base_freq*3/2, base_freq*15/8]
        elif 'm7' in quality:
            return [base_freq, base_freq*6/5, base_freq*3/2, base_freq*9/5]
        elif '7' in quality:
            return [base_freq, base_freq*5/4, base_freq*3/2, base_freq*9/5]
        elif 'min' in quality:
            return [base_freq, base_freq*6/5, base_freq*3/2]
        else:
            return [base_freq, base_freq*5/4, base_freq*3/2]

    def parse_natural_command(self, command: str) -> Tuple[str, str, int]:
        """Parse natural language command with improved understanding"""
        command = command.lower()
        
        # Extract style
        style = None
        max_style_matches = 0
        for s, keywords in self.style_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in command)
            if matches > max_style_matches:
                style = s
                max_style_matches = matches
        
        # Extract emotion
        emotion = None
        max_emotion_matches = 0
        for e, keywords in self.emotion_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in command)
            if matches > max_emotion_matches:
                if style and e in self.styles[style]['scales']:
                    emotion = e
                    max_emotion_matches = matches
        
        # Extract duration with improved time parsing
        duration = 60  # default duration
        
        # First check for explicit numbers with units
        time_patterns = [
            r'(\d+)\s*(?:minute|minutes|min|mins)',
            r'(\d+)\s*(?:second|seconds|sec|secs)',
            r'(\d+)\s*(?:hour|hours|hr|hrs)'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, command)
            if matches:
                number = int(matches[0])
                if 'hour' in pattern or 'hr' in pattern:
                    duration = min(number * 3600, 3600)  # Cap at 1 hour
                elif 'minute' in pattern or 'min' in pattern:
                    duration = min(number * 60, 3600)
                else:
                    duration = min(number, 3600)
                break
        
        # Check for quantity multipliers
        for word, multiplier in self.context_keywords['quantity_multipliers'].items():
            if word in command:
                duration = int(duration * multiplier)
                break
        
        # If no explicit duration found, check for duration keywords
        if duration == 60:
            for keyword, value in self.duration_keywords.items():
                if keyword in command:
                    duration = value
                    break
        
        # Apply constraints
        duration = max(10, min(3600, duration))
        
        # If style or emotion not found, use context clues
        if not style:
            if any(word in command for word in ['relax', 'calm', 'peace', 'chill', 'sleep']):
                style = 'ambient'
            elif any(word in command for word in ['party', 'dance', 'energy', 'club']):
                style = 'electronic'
            elif any(word in command for word in ['study', 'focus', 'work']):
                style = 'lofi'
            else:
                style = 'jazz'  # default
        
        if not emotion or emotion not in self.styles[style]['scales']:
            if 'calm' in command or 'relax' in command:
                emotion = 'peace' if 'peace' in self.styles[style]['scales'] else 'dreamy'
            elif 'energy' in command or 'power' in command:
                emotion = 'energy' if 'energy' in self.styles[style]['scales'] else 'intensity'
            elif 'dark' in command or 'night' in command:
                emotion = 'mystery' if 'mystery' in self.styles[style]['scales'] else 'noir'
            else:
                # Pick first available emotion for the style
                emotion = list(self.styles[style]['scales'].keys())[0]
        
        return style, emotion, duration

    def _add_drums(self, audio: np.ndarray, start: int, duration: float, style: str) -> None:
        """Add drum patterns based on style"""
        drum_duration = duration * 0.2
        samples = len(audio)
        
        # Get style-specific drum patterns
        if style == 'jazz':
            # Swing pattern
            if random.random() > 0.7:  # 30% chance of kick
                kick = self._generate_drum_sound('kick', drum_duration)
                if start + len(kick) <= samples:
                    audio[start:start + len(kick)] += kick * 0.4
            
            # Ride cymbal pattern
            hihat = self._generate_drum_sound('hihat', drum_duration)
            if start + len(hihat) <= samples:
                audio[start:start + len(hihat)] += hihat * 0.2
                
        elif style in ['electronic', 'hiphop']:
            # Four-on-the-floor pattern
            kick = self._generate_drum_sound('kick', drum_duration)
            if start + len(kick) <= samples:
                audio[start:start + len(kick)] += kick * 0.5
            
            # Snare on 2 and 4
            if random.random() > 0.5:
                snare = self._generate_drum_sound('snare', drum_duration)
                if start + len(snare) <= samples:
                    audio[start:start + len(snare)] += snare * 0.4
            
            # Hi-hat pattern
            hihat = self._generate_drum_sound('hihat', drum_duration)
            if start + len(hihat) <= samples:
                audio[start:start + len(hihat)] += hihat * 0.3
                
        elif style == 'rock':
            # Rock pattern
            if random.random() > 0.3:
                kick = self._generate_drum_sound('kick', drum_duration)
                if start + len(kick) <= samples:
                    audio[start:start + len(kick)] += kick * 0.6
            
            snare = self._generate_drum_sound('snare', drum_duration)
            if start + len(snare) <= samples:
                audio[start:start + len(snare)] += snare * 0.5
            
            hihat = self._generate_drum_sound('hihat', drum_duration)
            if start + len(hihat) <= samples:
                audio[start:start + len(hihat)] += hihat * 0.3
                
        else:  # ambient, classical
            # Subtle percussion
            if random.random() > 0.8:  # 20% chance of percussion
                hihat = self._generate_drum_sound('hihat', drum_duration)
                if start + len(hihat) <= samples:
                    audio[start:start + len(hihat)] += hihat * 0.1

    def _apply_effects(self, audio: np.ndarray, style: str, emotion: str) -> np.ndarray:
        """Apply improved audio effects with better quality and mastering"""
        # Apply EQ with style-specific curves
        if style in ['electronic', 'hiphop']:
            # Boost bass and highs for electronic/hiphop
            b, a = signal.butter(2, 100 / (self.sample_rate/2))
            bass = signal.filtfilt(b, a, audio)
            audio += bass * 0.6
            
            b, a = signal.butter(2, 8000 / (self.sample_rate/2), 'high')
            highs = signal.filtfilt(b, a, audio)
            audio += highs * 0.4
            
        elif style in ['jazz', 'classical']:
            # Enhance mids and warmth
            b, a = signal.butter(2, [300 / (self.sample_rate/2), 5000 / (self.sample_rate/2)], 'band')
            mids = signal.filtfilt(b, a, audio)
            audio += mids * 0.4
            
            # Add warmth
            b, a = signal.butter(2, 200 / (self.sample_rate/2))
            warmth = signal.filtfilt(b, a, audio)
            audio += warmth * 0.3
            
        elif style in ['rock', 'metal']:
            # Aggressive EQ for rock/metal
            b, a = signal.butter(2, 100 / (self.sample_rate/2))
            bass = signal.filtfilt(b, a, audio)
            audio += bass * 0.7
            
            b, a = signal.butter(2, [2000 / (self.sample_rate/2), 5000 / (self.sample_rate/2)], 'band')
            mids = signal.filtfilt(b, a, audio)
            audio += mids * 0.5
            
        elif style in ['ambient', 'lofi']:
            # Soft EQ for ambient/lofi
            b, a = signal.butter(2, 200 / (self.sample_rate/2))
            bass = signal.filtfilt(b, a, audio)
            audio += bass * 0.3
            
            b, a = signal.butter(2, [1000 / (self.sample_rate/2), 4000 / (self.sample_rate/2)], 'band')
            mids = signal.filtfilt(b, a, audio)
            audio += mids * 0.2
            
        # Apply dynamic compression with style-specific settings
        if style in ['electronic', 'hiphop', 'rock']:
            threshold = 0.4
            ratio = 4
            attack = 0.01
            release = 0.1
        elif style in ['jazz', 'classical']:
            threshold = 0.3
            ratio = 2
            attack = 0.05
            release = 0.2
        else:
            threshold = 0.35
            ratio = 3
            attack = 0.03
            release = 0.15
        
        # Improved compression with soft knee
        def compress(x, threshold, ratio, attack, release, knee=0.1):
            gain = np.ones_like(x)
            mask = np.abs(x) > threshold
            soft_mask = (np.abs(x) > (threshold - knee)) & (np.abs(x) <= (threshold + knee))
            
            # Apply attack and release
            attack_samples = int(attack * self.sample_rate)
            release_samples = int(release * self.sample_rate)
            
            # Calculate gain reduction
            gain[mask] = (1 + (1/ratio - 1) * (np.abs(x[mask]) - threshold) / threshold)
            
            # Soft knee transition
            soft_gain = 1 + (1/ratio - 1) * (np.abs(x[soft_mask]) - (threshold - knee)) / (2 * knee)
            gain[soft_mask] = soft_gain
            
            # Apply attack and release
            gain = np.convolve(gain, np.ones(attack_samples)/attack_samples, mode='same')
            gain = np.convolve(gain, np.ones(release_samples)/release_samples, mode='same')
            
            return x * gain
        
        audio = compress(audio, threshold, ratio, attack, release)
        
        # Apply reverb with style-specific settings
        if style in ['ambient', 'classical']:
            reverb_length = int(2.0 * self.sample_rate)  # 2 seconds reverb
            reverb = np.exp(-2 * np.linspace(0, 1, reverb_length))
        elif style in ['rock', 'metal']:
            reverb_length = int(1.0 * self.sample_rate)  # 1 second reverb
            reverb = np.exp(-3 * np.linspace(0, 1, reverb_length))
        else:
            reverb_length = int(1.5 * self.sample_rate)  # 1.5 seconds reverb
            reverb = np.exp(-2.5 * np.linspace(0, 1, reverb_length))
        
        audio = signal.convolve(audio, reverb, mode='same')
        
        # Apply stereo widening for appropriate styles
        if style in ['electronic', 'rock', 'synthwave']:
            # Enhanced stereo widening
            audio_shifted = np.roll(audio, 30)  # Increased delay
            audio = np.stack([audio, audio_shifted])
            # Add subtle phase variation
            phase = np.sin(2 * np.pi * 0.1 * np.arange(len(audio[0])) / self.sample_rate)
            audio[1] *= (1 + 0.1 * phase)
            audio = audio.mean(axis=0)
        
        # Apply final limiting and saturation
        if style in ['electronic', 'hiphop', 'rock']:
            # More aggressive limiting
            audio = np.tanh(audio * 1.2)
        else:
            # Gentler limiting
            audio = np.tanh(audio * 0.9)
        
        # Apply subtle noise reduction
        if style in ['ambient', 'lofi']:
            noise_floor = 0.0001
            audio = np.where(np.abs(audio) < noise_floor, 0, audio)
        
        return audio

    def generate_music(self, style: str, emotion: str, duration: int = 30) -> Optional[Tuple[np.ndarray, int]]:
        """Generate music with separate instrument tracks and unique beats"""
        try:
            print("\nInitializing music generation...")
            
            # Validate input parameters
            if style not in self.styles:
                raise ValueError(f"Invalid style: {style}")
            if emotion not in self.emotion_keywords:
                raise ValueError(f"Invalid emotion: {emotion}")
            if duration <= 0 or duration > 600:  # Max 10 minutes
                raise ValueError("Duration must be between 1 and 600 seconds")
            
            # Initialize audio buffer for each instrument
            total_samples = int(duration * self.sample_rate)
            tracks = {}
            
            # Get style parameters
            style_params = self.styles[style]
            tempo = style_params['tempo']
            beat_duration = 60 / tempo
            progression = random.choice(style_params['progressions'])
            scale = style_params['scales'][emotion]
            
            # Initialize tracks based on style's instruments
            for instrument in style_params['instruments']:
                tracks[instrument] = np.zeros(total_samples, dtype=np.float32)  # Use float32 for memory efficiency
            
            # Calculate total steps for progress bar
            total_steps = int(duration / beat_duration)
            
            print(f"\nCreating {emotion} {style} music...")
            print(f"Style: {style.capitalize()} (Tempo: {tempo} BPM)")
            print(f"Instruments: {', '.join(style_params['instruments'])}")
            
            # Create rhythm patterns for each instrument
            rhythm_patterns = self._create_rhythm_patterns(style, emotion, total_steps)
            
            with tqdm(total=total_steps, desc="Generating music", unit="beat") as pbar:
                # Generate continuous music
                for beat in range(total_steps):
                    try:
                        beat_start = int(beat * beat_duration * self.sample_rate)
                        beat_end = min(beat_start + int(beat_duration * self.sample_rate), total_samples)
                        beat_length = beat_end - beat_start
                        
                        # Determine current chord (cycle through progression)
                        chord_idx = (beat // 4) % len(progression)
                        chord = progression[chord_idx]
                        
                        # Generate piano/synth track with rhythm pattern
                        if 'piano' in style_params['instruments']:
                            if rhythm_patterns['piano'][beat]:
                                frequencies = self._get_chord_notes(chord)
                                for freq in frequencies:
                                    note = self._generate_note(freq, beat_duration * 4, self.instruments['piano'])
                                    note = note[:beat_length]  # Ensure note fits in beat
                                    if beat_start + len(note) <= total_samples:
                                        tracks['piano'][beat_start:beat_start + len(note)] += note * 0.3
                        
                        # Generate saxophone track with rhythm pattern
                        if 'saxophone' in style_params['instruments']:
                            if rhythm_patterns['saxophone'][beat]:
                                note_duration = beat_duration * random.choice([0.5, 1.0, 1.5, 2.0])
                                note_freq = self.note_frequencies[random.choice(scale)]
                                note = self._generate_note(note_freq, note_duration, self.instruments['saxophone'])
                                note = note[:beat_length]  # Ensure note fits in beat
                                if beat_start + len(note) <= total_samples:
                                    tracks['saxophone'][beat_start:beat_start + len(note)] += note * 0.3
                        
                        # Generate bass track with rhythm pattern
                        if 'bass' in style_params['instruments']:
                            if rhythm_patterns['bass'][beat]:
                                frequencies = self._get_chord_notes(chord)
                                bass_freq = frequencies[0] / 2  # Lower octave
                                note = self._generate_note(bass_freq, beat_duration * 4, self.instruments['bass'])
                                note = note[:beat_length]  # Ensure note fits in beat
                                if beat_start + len(note) <= total_samples:
                                    tracks['bass'][beat_start:beat_start + len(note)] += note * 0.4
                        
                        # Generate drum track with enhanced patterns
                        if 'drums' in style_params['instruments']:
                            pattern = self._generate_drum_pattern(style, beat % 4)
                            for drum_type, should_play in pattern.items():
                                if should_play:
                                    drum_sound = self._generate_drum_sound(drum_type, beat_duration)
                                    drum_sound = drum_sound[:beat_length]  # Ensure sound fits in beat
                                    if beat_start + len(drum_sound) <= total_samples:
                                        tracks['drums'][beat_start:beat_start + len(drum_sound)] += drum_sound * 0.5
                        
                        # Generate synth track with rhythm pattern
                        if 'synth' in style_params['instruments']:
                            if rhythm_patterns['synth'][beat]:
                                pad_freq = self.note_frequencies[random.choice(scale)]
                                note = self._generate_note(pad_freq, beat_duration * 8, self.instruments['synth'])
                                note = note[:beat_length]  # Ensure note fits in beat
                                if beat_start + len(note) <= total_samples:
                                    tracks['synth'][beat_start:beat_start + len(note)] += note * 0.2
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"Warning: Error in beat {beat}: {str(e)}")
                        continue
            
            print("\nApplying effects and mastering...")
            # Apply effects to each track
            processed_tracks = {}
            for instrument, track in tracks.items():
                try:
                    processed_tracks[instrument] = self._apply_effects(track, style, emotion)
                    # Clear original track to free memory
                    tracks[instrument] = None
                except Exception as e:
                    print(f"Warning: Error processing {instrument} track: {str(e)}")
                    processed_tracks[instrument] = track  # Use original track if processing fails
            
            # Mix tracks with style-specific balance
            final_audio = np.zeros(total_samples, dtype=np.float32)
            
            # Mix tracks
            for instrument, track in processed_tracks.items():
                try:
                    level = style_params['mix_levels'].get(instrument, 0.3)
                    final_audio += track * level
                    # Clear processed track to free memory
                    processed_tracks[instrument] = None
                except Exception as e:
                    print(f"Warning: Error mixing {instrument} track: {str(e)}")
            
            print("Finalizing audio...")
            # Final normalization
            try:
                max_val = np.max(np.abs(final_audio))
                if max_val > 0:
                    final_audio = final_audio / max_val * 0.9
            except Exception as e:
                print(f"Warning: Error in final normalization: {str(e)}")
            
            # Save individual tracks for reference
            try:
                if not os.path.exists('generated_music'):
                    os.makedirs('generated_music')
                
                for instrument, track in processed_tracks.items():
                    if track is not None:  # Only save if track wasn't cleared
                        track_path = f'generated_music/{style}_{emotion}_{instrument}.wav'
                        sf.write(track_path, track, self.sample_rate)
            except Exception as e:
                print(f"Warning: Error saving tracks: {str(e)}")
            
            return final_audio, self.sample_rate
            
        except Exception as e:
            print(f"Error generating music: {e}")
            return None

    def _create_rhythm_patterns(self, style: str, emotion: str, total_steps: int) -> Dict[str, np.ndarray]:
        """Create sophisticated rhythm patterns for each instrument"""
        patterns = {}
        
        # Define base patterns for different styles
        if style == 'jazz':
            patterns['piano'] = np.array([1, 0, 1, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['saxophone'] = np.array([0, 1, 0, 1] * (total_steps // 4 + 1))[:total_steps]
            patterns['bass'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['synth'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
        elif style == 'electronic':
            patterns['piano'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['bass'] = np.array([1, 0, 1, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['synth'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
        elif style == 'rock':
            patterns['guitar'] = np.array([1, 0, 1, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['bass'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['drums'] = np.array([1, 1, 1, 1] * (total_steps // 4 + 1))[:total_steps]
        elif style == 'ambient':
            patterns['synth'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['piano'] = np.array([0, 0, 1, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['strings'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
        elif style == 'hiphop':
            patterns['synth'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['bass'] = np.array([1, 0, 1, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['drums'] = np.array([1, 1, 1, 1] * (total_steps // 4 + 1))[:total_steps]
        elif style == 'classical':
            patterns['piano'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['strings'] = np.array([0, 1, 0, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['harp'] = np.array([0, 0, 1, 0] * (total_steps // 4 + 1))[:total_steps]
        elif style == 'lofi':
            patterns['piano'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['bass'] = np.array([1, 0, 1, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['drums'] = np.array([1, 1, 1, 1] * (total_steps // 4 + 1))[:total_steps]
        elif style == 'synthwave':
            patterns['synth'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['bass'] = np.array([1, 0, 1, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['drums'] = np.array([1, 1, 1, 1] * (total_steps // 4 + 1))[:total_steps]
        elif style == 'folk':
            patterns['guitar'] = np.array([1, 0, 1, 0] * (total_steps // 4 + 1))[:total_steps]
            patterns['violin'] = np.array([0, 1, 0, 1] * (total_steps // 4 + 1))[:total_steps]
            patterns['bass'] = np.array([1, 0, 0, 0] * (total_steps // 4 + 1))[:total_steps]
        
        # Add variations based on emotion
        for instrument in patterns:
            if emotion == 'energy':
                # More frequent notes
                patterns[instrument] = np.where(patterns[instrument] == 0, 
                                              np.random.random(len(patterns[instrument])) > 0.7,
                                              patterns[instrument])
            elif emotion == 'calmness':
                # Less frequent notes
                patterns[instrument] = np.where(patterns[instrument] == 1,
                                              np.random.random(len(patterns[instrument])) > 0.3,
                                              patterns[instrument])
            elif emotion == 'mystery':
                # More syncopation
                patterns[instrument] = np.roll(patterns[instrument], 1)
        
        return patterns

    def _generate_drum_pattern(self, style: str, beat_position: int) -> Dict[str, bool]:
        """Generate unique drum patterns with enhanced variations"""
        base_patterns = {
            'jazz': [
                {'kick': True, 'snare': False, 'hihat': True, 'tom': False, 'clap': False, 'ride': True},  # 1
                {'kick': False, 'snare': False, 'hihat': True, 'tom': True, 'clap': False, 'ride': False},  # 2
                {'kick': False, 'snare': True, 'hihat': True, 'tom': False, 'clap': True, 'ride': True},   # 3
                {'kick': False, 'snare': False, 'hihat': True, 'tom': True, 'clap': False, 'ride': False}   # 4
            ],
            'electronic': [
                {'kick': True, 'snare': False, 'hihat': True, 'tom': False, 'clap': True, 'ride': False},   # 1
                {'kick': False, 'snare': False, 'hihat': True, 'tom': False, 'clap': False, 'ride': False}, # 2
                {'kick': False, 'snare': True, 'hihat': True, 'tom': False, 'clap': True, 'ride': False},   # 3
                {'kick': False, 'snare': False, 'hihat': True, 'tom': False, 'clap': False, 'ride': False}  # 4
            ],
            'rock': [
                {'kick': True, 'snare': False, 'hihat': True, 'tom': False, 'clap': False, 'ride': False},  # 1
                {'kick': False, 'snare': True, 'hihat': True, 'tom': True, 'clap': False, 'ride': False},   # 2
                {'kick': True, 'snare': False, 'hihat': True, 'tom': False, 'clap': False, 'ride': False},  # 3
                {'kick': False, 'snare': True, 'hihat': True, 'tom': True, 'clap': False, 'ride': False}    # 4
            ],
            'hiphop': [
                {'kick': True, 'snare': False, 'hihat': True, 'tom': False, 'clap': True, 'ride': False},   # 1
                {'kick': False, 'snare': True, 'hihat': True, 'tom': False, 'clap': False, 'ride': False},  # 2
                {'kick': True, 'snare': False, 'hihat': True, 'tom': False, 'clap': True, 'ride': False},   # 3
                {'kick': False, 'snare': True, 'hihat': True, 'tom': False, 'clap': False, 'ride': False}   # 4
            ],
            'ambient': [
                {'kick': False, 'snare': False, 'hihat': False, 'tom': True, 'clap': False, 'ride': False}, # 1
                {'kick': False, 'snare': False, 'hihat': False, 'tom': False, 'clap': False, 'ride': False},# 2
                {'kick': False, 'snare': False, 'hihat': False, 'tom': True, 'clap': False, 'ride': False}, # 3
                {'kick': False, 'snare': False, 'hihat': False, 'tom': False, 'clap': False, 'ride': False} # 4
            ],
            'classical': [
                {'kick': False, 'snare': False, 'hihat': False, 'tom': True, 'clap': False, 'ride': False}, # 1
                {'kick': False, 'snare': False, 'hihat': False, 'tom': False, 'clap': False, 'ride': False},# 2
                {'kick': False, 'snare': False, 'hihat': False, 'tom': True, 'clap': False, 'ride': False}, # 3
                {'kick': False, 'snare': False, 'hihat': False, 'tom': False, 'clap': False, 'ride': False} # 4
            ],
            'lofi': [
                {'kick': True, 'snare': False, 'hihat': True, 'tom': False, 'clap': False, 'ride': False},  # 1
                {'kick': False, 'snare': True, 'hihat': True, 'tom': False, 'clap': False, 'ride': False},  # 2
                {'kick': True, 'snare': False, 'hihat': True, 'tom': False, 'clap': False, 'ride': False},  # 3
                {'kick': False, 'snare': True, 'hihat': True, 'tom': False, 'clap': False, 'ride': False}   # 4
            ],
            'synthwave': [
                {'kick': True, 'snare': False, 'hihat': True, 'tom': False, 'clap': True, 'ride': False},   # 1
                {'kick': False, 'snare': False, 'hihat': True, 'tom': False, 'clap': False, 'ride': False}, # 2
                {'kick': False, 'snare': True, 'hihat': True, 'tom': False, 'clap': True, 'ride': False},   # 3
                {'kick': False, 'snare': False, 'hihat': True, 'tom': False, 'clap': False, 'ride': False}  # 4
            ],
            'folk': [
                {'kick': True, 'snare': False, 'hihat': False, 'tom': True, 'clap': False, 'ride': False},  # 1
                {'kick': False, 'snare': True, 'hihat': False, 'tom': False, 'clap': False, 'ride': False}, # 2
                {'kick': True, 'snare': False, 'hihat': False, 'tom': True, 'clap': False, 'ride': False},  # 3
                {'kick': False, 'snare': True, 'hihat': False, 'tom': False, 'clap': False, 'ride': False}  # 4
            ]
        }
        
        # Get base pattern for style or default to jazz
        patterns = base_patterns.get(style, base_patterns['jazz'])
        pattern = patterns[beat_position].copy()
        
        # Add style-specific variations
        if style == 'jazz':
            # Add swing feel and ghost notes
            if beat_position in [1, 3]:
                pattern['hihat'] = random.random() > 0.3
                if random.random() > 0.7:
                    pattern['snare'] = True
                if random.random() > 0.5:
                    pattern['ride'] = not pattern['ride']
        elif style == 'electronic':
            # Add extra hi-hats and claps for energy
            if random.random() > 0.6:
                pattern['hihat'] = True
            if random.random() > 0.8:
                pattern['clap'] = True
            # Add occasional tom fills
            if random.random() > 0.9:
                pattern['tom'] = True
        elif style == 'hiphop':
            # Add trap-style hi-hat rolls and syncopation
            if random.random() > 0.7:
                pattern['hihat'] = True
            if random.random() > 0.5:
                pattern['clap'] = not pattern['clap']
            # Add occasional tom rolls
            if random.random() > 0.8:
                pattern['tom'] = True
        elif style == 'ambient':
            # Add subtle percussion variations
            if random.random() > 0.8:
                pattern['tom'] = True
            # Add occasional ride cymbal
            if random.random() > 0.9:
                pattern['ride'] = True
        elif style == 'classical':
            # Add orchestral percussion variations
            if random.random() > 0.9:
                pattern['tom'] = True
            # Add occasional ride cymbal
            if random.random() > 0.8:
                pattern['ride'] = True
        elif style == 'lofi':
            # Add vinyl-style variations
            if random.random() > 0.7:
                pattern['hihat'] = not pattern['hihat']
            # Add occasional clap
            if random.random() > 0.6:
                pattern['clap'] = True
        elif style == 'synthwave':
            # Add retro-style variations
            if random.random() > 0.6:
                pattern['clap'] = True
            # Add occasional ride cymbal
            if random.random() > 0.7:
                pattern['ride'] = True
        elif style == 'folk':
            # Add traditional percussion variations
            if random.random() > 0.7:
                pattern['tom'] = True
            # Add occasional clap
            if random.random() > 0.6:
                pattern['clap'] = True
        
        # Add random variations to keep it interesting
        if random.random() > 0.7:  # 30% chance of variation
            if random.random() > 0.5:
                pattern['kick'] = not pattern['kick']
            if random.random() > 0.5:
                pattern['snare'] = not pattern['snare']
            if random.random() > 0.5:
                pattern['hihat'] = not pattern['hihat']
            if random.random() > 0.5:
                pattern['tom'] = not pattern['tom']
            if random.random() > 0.5:
                pattern['clap'] = not pattern['clap']
            if random.random() > 0.5:
                pattern['ride'] = not pattern['ride']
        
        return pattern

# FastAPI setup
app = FastAPI()
generator = MusicGenerator()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MusicRequest(BaseModel):
    prompt: str
    duration: Optional[int] = None

@app.get("/")
async def root():
    return {
        "message": "Welcome to the AI Music Generator API",
        "endpoints": {
            "/generate": "Generate music from text prompt",
            "/styles": "List available music styles",
            "/emotions": "List available emotions"
        }
    }

@app.get("/styles")
async def get_styles():
    return {
        "styles": list(generator.styles.keys()),
        "description": "Available music styles for generation"
    }

@app.get("/emotions")
async def get_emotions():
    return {
        "emotions": list(generator.styles['jazz']['scales'].keys()),
        "description": "Available emotions for music generation"
    }

@app.post("/generate")
async def generate_music(request: MusicRequest):
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists('generated_music'):
            os.makedirs('generated_music')
        
        # Generate music from text prompt
        result = generator.generate_music(request.prompt)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to generate music")
        
        audio, sr = result
        
        # Save the generated music
        output_file = f'generated_music/generated_music.wav'
        sf.write(output_file, audio, sr)
        
        return FileResponse(
            output_file,
            media_type='audio/wav',
            filename='generated_music.wav'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_user_input():
    """Get user preferences using natural language"""
    print("\n=== AI Music Generator ===")
    print("\nTell me what kind of music you'd like! You can say something like:")
    print("- 'Create a relaxing ambient piece for 2 minutes'")
    print("- 'Generate some energetic electronic music'")
    print("- 'Make a mysterious jazz song'")
    print("- 'I want dramatic classical music for 90 seconds'")
    
    while True:
        try:
            command = input("\nWhat would you like me to create? ").strip()
            if command.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                exit(0)
            
            if len(command) < 3:
                print("Please provide more details about the music you want.")
                continue
                
            generator = MusicGenerator()
            style, emotion, duration = generator.parse_natural_command(command)
            
            print(f"\nI'll create {emotion} {style} music for {duration} seconds.")
            confirm = input("Sound good? (yes/no): ").lower().strip()
            
            if confirm.startswith('y'):
                return generator, style, emotion, duration
            else:
                print("\nOk, let's try again!")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            exit(0)
        except Exception as e:
            print(f"I didn't quite understand that. Could you try rephrasing?")

def main():
    """Generate and save music based on natural language input"""
    try:
        while True:
            generator, style, emotion, duration = get_user_input()
            
            result = generator.generate_music(style=style, emotion=emotion, duration=duration)
            
            if result:
                audio, sr = result
                # Save the generated music
                output_file = f'generated_music/{style}_{emotion}.wav'
                if not os.path.exists('generated_music'):
                    os.makedirs('generated_music')
                    
                print(f"\nSaving your music...")
                sf.write(output_file, audio, sr)
                print(f"\nDone! Your music has been saved to {output_file}")
                
                print("\nWould you like to create another piece? (yes/no)")
                if not input().lower().strip().startswith('y'):
                    print("\nGoodbye! Enjoy your music!")
                    break
            else:
                print("\nSorry, there was a problem generating the music.")
                print("Would you like to try again? (yes/no)")
                if not input().lower().strip().startswith('y'):
                    print("\nGoodbye!")
                    break
                    
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Would you like to try again? (yes/no)")
        if input().lower().strip().startswith('y'):
            main()

if __name__ == "__main__":
    main() 