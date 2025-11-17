import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import shutil # Pour vérifier FFmpeg

# Importer notre logique métier depuis l'autre fichier
from processor import VideoProcessor

class CutGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoEditor Video - Pro Highlight Generator")
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar(value="highlight.mp4") # Fichier de sortie
        self.profile_var = tk.StringVar(value="Moyen") # Profil
        self.tiktok_var = tk.BooleanVar(value=True) # Checkbox Tiktok
        
        # --- Configuration de l'interface ---
        
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configuration des colonnes et lignes pour le redimensionnement
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1) # Colonne des 'Entry' s'étend
        main_frame.rowconfigure(5, weight=1) # Ligne du 'Text' s'étend

        # Ligne 1: Fichier d'entrée
        ttk.Label(main_frame, text="Fichier Stream:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_file, width=60).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(main_frame, text="Parcourir...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)
        
        # Ligne 2: Fichier de sortie (Highlight)
        ttk.Label(main_frame, text="Fichier Highlight:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_file, width=60).grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(main_frame, text="Enregistrer...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Ligne 3: Profil de montage
        ttk.Label(main_frame, text="Profil Highlight:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        profile_combo = ttk.Combobox(main_frame, textvariable=self.profile_var, 
                                     values=["Court", "Moyen", "Longue"], state="readonly")
        profile_combo.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        
        # Ligne 4: Options (Tiktok)
        tiktok_check = ttk.Checkbutton(main_frame, 
                                       text="Générer aussi les clips Tiktok (9:16)", 
                                       variable=self.tiktok_var)
        tiktok_check.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # Ligne 5: Barre de progression
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew")
        
        self.progress_label = ttk.Label(progress_frame, text="0.0%", width=6, anchor="e")
        self.progress_label.grid(row=0, column=1, sticky="e", padx=5)
        
        # Ligne 6: Zone de log
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=15, width=70, state="disabled", wrap="word")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Ligne 7: Bouton Démarrer
        self.start_button = ttk.Button(main_frame, text="Démarrer", command=self.start_process)
        self.start_button.grid(row=6, column=1, pady=10)
    
    def browse_input(self):
        filename = filedialog.askopenfilename(title="Sélectionnez le fichier vidéo",
                                              filetypes=[("Video Files", "*.mp4 *.mkv *.mov *.avi *.flv *.webm *.wmv"), ("All Files", "*.*")])
        if filename:
            self.input_file.set(filename)
            
            # Suggérer un nom de fichier de sortie
            base_name = os.path.splitext(os.path.basename(filename))[0]
            base_dir = os.path.dirname(filename)
            self.output_file.set(os.path.join(base_dir, f"{base_name}_highlight.mp4"))
    
    def browse_output(self):
        filename = filedialog.asksaveasfilename(title="Enregistrer le fichier de sortie",
                                                defaultextension=".mp4",
                                                filetypes=[("MP4", "*.mp4")])
        if filename:
            self.output_file.set(filename)

    def log(self, message):
        """Callback pour afficher les logs dans le Text widget."""
        self.root.after(0, self._insert_log, message)
        
    def _insert_log(self, message):
        """Méthode interne pour l'insertion de log (appelée par self.log)."""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
    
    def _clear_log(self):
        """Méthode interne pour vider la zone de log."""
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")
    
    def update_progress(self, value):
        """Callback pour mettre à jour la barre de progression."""
        self.root.after(0, self._set_progress, value)
        
    def _set_progress(self, value):
        """Méthode interne pour la progression (appelée par update_progress)."""
        self.progress['value'] = value
        self.progress_label.config(text=f"{value:.1f}%")
    
    def start_process(self):
        """
        Démarre le processus de traitement dans un thread séparé
        pour ne pas geler l'interface.
        """
        in_file = self.input_file.get()
        out_file = self.output_file.get() # Fichier Highlight
        profile = self.profile_var.get()
        generate_tiktoks = self.tiktok_var.get()
        
        if not in_file or not out_file:
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier d'entrée et un fichier de sortie.")
            return
            
        if not os.path.exists(in_file):
            messagebox.showerror("Erreur", "Le fichier d'entrée n'existe pas.")
            return
            
        self.start_button.config(state="disabled")
        self.progress['value'] = 0
        self.progress_label.config(text="0.0%")
        self._clear_log()
        
        # Créer l'instance du processeur
        processor = VideoProcessor(
            input_file=in_file,
            output_file=out_file,
            log_callback=self.log,
            progress_callback=self.update_progress,
            profile=profile,
            generate_tiktoks=generate_tiktoks
        )
        
        # Démarrer le processus dans un thread
        threading.Thread(target=self.run_process_thread, args=(processor,), daemon=True).start()
        
    def run_process_thread(self, processor):
        """
        Wrapper pour le thread qui exécute le processeur.
        Gère les erreurs et réactive le bouton à la fin.
        """
        try:
            processor.process()
        except Exception as e:
            # Afficher l'erreur dans l'UI
            self.root.after(0, lambda e=e: messagebox.showerror("Erreur de traitement", f"Une erreur est survenue:\n{e}"))
        finally:
            # Réactiver le bouton, quoi qu'il arrive
            self.root.after(0, lambda: self.start_button.config(state="normal"))

def setup_dark_theme(root):
    """Configure un thème sombre complet pour l'application."""
    
    BG_COLOR = "#2b2b2b"
    FG_COLOR = "#dcdcdc"
    ACCENT_COLOR = "#007acc"
    ENTRY_BG = "#3c3c3c"
    BTN_FG = "#ffffff"

    root.configure(bg=BG_COLOR)
    
    style = ttk.Style(root)
    style.theme_use('clam')

    # Styles globaux
    style.configure('.',
                    background=BG_COLOR,
                    foreground=FG_COLOR,
                    fieldbackground=ENTRY_BG,
                    troughcolor=BG_COLOR)
    
    # Style TFrame
    style.configure('TFrame', background=BG_COLOR)

    # Style TLabel
    style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR)

    # Style TButton
    style.configure('TButton',
                    background=ACCENT_COLOR,
                    foreground=BTN_FG,
                    padding=5,
                    relief='flat',
                    font=('TkDefaultFont', 10, 'bold'))
    style.map('TButton',
              background=[('active', '#005fae'), ('disabled', '#555555')],
              foreground=[('disabled', '#888888')])

    # Style TEntry
    style.configure('TEntry',
                    fieldbackground=ENTRY_BG,
                    foreground=FG_COLOR,
                    insertcolor=FG_COLOR,
                    borderwidth=1,
                    relief='flat')
    style.map('TEntry',
              bordercolor=[('focus', ACCENT_COLOR)],
              fieldbackground=[('disabled', '#444444')])

    # Style TProgressbar
    style.configure('Horizontal.TProgressbar',
                    background=ACCENT_COLOR,
                    troughcolor=ENTRY_BG,
                    thickness=10,
                    borderwidth=0)

    # Style TScrollbar
    style.configure('Vertical.TScrollbar',
                    gripcount=0,
                    background=ACCENT_COLOR,
                    troughcolor=ENTRY_BG,
                    arrowcolor=FG_COLOR,
                    relief='flat')
    style.map('Vertical.TScrollbar',
              background=[('active', '#005fae')])
              
    # Style TCombobox
    style.configure('TCombobox', 
                    fieldbackground=ENTRY_BG, 
                    foreground=FG_COLOR,
                    background=ENTRY_BG,
                    arrowcolor=FG_COLOR)
    style.map('TCombobox', 
              fieldbackground=[('readonly', ENTRY_BG)],
              foreground=[('readonly', FG_COLOR)])

    # Style TCheckbutton
    style.configure('TCheckbutton',
                    background=BG_COLOR,
                    foreground=FG_COLOR,
                    indicatorcolor=ENTRY_BG,
                    padding=5)
    style.map('TCheckbutton',
              indicatorcolor=[('selected', ACCENT_COLOR)])


def check_ffmpeg_tools():
    """
    Vérifie si FFmpeg et FFprobe sont installés et accessibles dans le PATH.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        messagebox.showerror("Erreur critique", 
            "FFmpeg et/ou FFprobe n'ont pas été trouvés.\n"
            "Veuillez les installer et vous assurer qu'ils sont\n"
            "dans le PATH de votre système pour que l'application fonctionne.")
        return False
    return True

if __name__ == "__main__":
    if check_ffmpeg_tools():
        root = tk.Tk()
        
        # Appliquer notre nouveau thème sombre
        setup_dark_theme(root)
            
        app = CutGUI(root)
        root.mainloop()
