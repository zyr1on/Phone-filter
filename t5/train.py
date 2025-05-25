import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

class PhoneDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=512):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = "Telefon önerisi: " + str(self.inputs[idx])
        target_text = str(self.targets[idx])
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.steps = []
    
    def log(self, logs):
        super().log(logs)
        
        # Save training metrics
        if 'loss' in logs:
            self.train_losses.append(logs['loss'])
            self.steps.append(self.state.global_step)
        
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])
        
        if 'learning_rate' in logs:
            self.learning_rates.append(logs['learning_rate'])

class PhoneT5Model:
    def __init__(self, model_name='t5-small'):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.trainer = None
        self.training_history = {}
        
        # Add special tokens if needed
        special_tokens = {
            'additional_special_tokens': ['<phone>', '<feature>', '<value>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def load_data(self, data_file):
        inputs = []
        targets = []
        
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if ' -> ' in line:
                input_text, output_text = line.split(' -> ', 1)
                inputs.append(input_text.strip())
                targets.append(output_text.strip())
        
        return inputs, targets
    
    def train(self, data_file, output_dir='./phone_t5_model', epochs=3, batch_size=8, learning_rate=5e-4):
        print("Veri yükleniyor...")
        inputs, targets = self.load_data(data_file)
        
        print(f"Toplam {len(inputs)} örnek yüklendi.")
        
        # Train-validation split
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            inputs, targets, test_size=0.2, random_state=42
        )
        
        print(f"Eğitim: {len(train_inputs)}, Doğrulama: {len(val_inputs)} örnek")
        
        # Create datasets
        train_dataset = PhoneDataset(train_inputs, train_targets, self.tokenizer)
        val_dataset = PhoneDataset(val_inputs, val_targets, self.tokenizer)
        
        # Training arguments with better monitoring
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=5,  # Daha sık log
            logging_strategy='steps',
            eval_strategy='steps',
            eval_steps=25,  # Daha sık evaluation
            save_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            report_to="none",
            dataloader_num_workers=0,
            
            lr_scheduler_type='cosine',
            warmup_ratio=0.1,
            fp16=torch.cuda.is_available(),
            # Early stopping
            early_stopping_patience=3,
            # Gradient clipping
            max_grad_norm=1.0
        )
        
        # Create custom trainer
        self.trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )
        
        print("Model eğitimi başlıyor...")
        print(f"Öğrenme oranı: {learning_rate}")
        print(f"Batch boyutu: {batch_size}")
        print(f"Epoch sayısı: {epochs}")
        print(f"GPU kullanımı: {'Evet' if torch.cuda.is_available() else 'Hayır'}")
        print("-" * 50)
        
        # Start training
        train_result = self.trainer.train()
        
        # Save training history
        self.training_history = {
            'train_losses': self.trainer.train_losses,
            'eval_losses': self.trainer.eval_losses,
            'learning_rates': self.trainer.learning_rates,
            'steps': self.trainer.steps,
            'train_runtime': train_result.metrics['train_runtime'],
            'train_samples_per_second': train_result.metrics['train_samples_per_second'],
            'final_train_loss': train_result.metrics['train_loss'],
            'model_name': self.model_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model
        print("Model kaydediliyor...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Plot training curves
        #self.plot_training_curves(output_dir)
        
        print(f"Model {output_dir} klasörüne kaydedildi!")
        print(f"Final eğitim kaybı: {train_result.metrics['train_loss']:.6f}")
        print(f"Eğitim süresi: {train_result.metrics['train_runtime']:.2f} saniye")
        print(f"Saniyede işlenen örnek: {train_result.metrics['train_samples_per_second']:.2f}")
        
        return train_result
    
    def plot_training_curves(self, output_dir):
        """Plot training and validation loss curves"""
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Training Loss
            if self.training_history['train_losses']:
                steps = self.training_history['steps'][:len(self.training_history['train_losses'])]
                ax1.plot(steps, self.training_history['train_losses'], 'b-', linewidth=2, label='Training Loss')
                ax1.set_xlabel('Steps')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss Over Time')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Add final loss text
                final_loss = self.training_history['train_losses'][-1]
                ax1.text(0.02, 0.98, f'Final Loss: {final_loss:.6f}', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot 2: Validation Loss
            if self.training_history['eval_losses']:
                eval_steps = np.linspace(0, max(self.training_history['steps']), len(self.training_history['eval_losses']))
                ax2.plot(eval_steps, self.training_history['eval_losses'], 'r-', linewidth=2, label='Validation Loss')
                ax2.set_xlabel('Steps')
                ax2.set_ylabel('Loss')
                ax2.set_title('Validation Loss Over Time')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Add best validation loss
                best_val_loss = min(self.training_history['eval_losses'])
                ax2.text(0.02, 0.98, f'Best Val Loss: {best_val_loss:.6f}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Plot 3: Learning Rate Schedule
            if self.training_history['learning_rates']:
                lr_steps = self.training_history['steps'][:len(self.training_history['learning_rates'])]
                ax3.plot(lr_steps, self.training_history['learning_rates'], 'g-', linewidth=2)
                ax3.set_xlabel('Steps')
                ax3.set_ylabel('Learning Rate')
                ax3.set_title('Learning Rate Schedule')
                ax3.grid(True, alpha=0.3)
                ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # Plot 4: Combined Loss Comparison
            if self.training_history['train_losses'] and self.training_history['eval_losses']:
                # Interpolate training loss to match eval steps
                eval_steps = np.linspace(0, max(self.training_history['steps']), len(self.training_history['eval_losses']))
                train_steps = self.training_history['steps'][:len(self.training_history['train_losses'])]
                
                ax4.plot(train_steps, self.training_history['train_losses'], 'b-', alpha=0.7, label='Training Loss')
                ax4.plot(eval_steps, self.training_history['eval_losses'], 'r-', linewidth=2, label='Validation Loss')
                ax4.set_xlabel('Steps')
                ax4.set_ylabel('Loss')
                ax4.set_title('Training vs Validation Loss')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
                
                # Check for overfitting
                if len(self.training_history['eval_losses']) > 5:
                    recent_val_trend = np.polyfit(range(5), self.training_history['eval_losses'][-5:], 1)[0]
                    if recent_val_trend > 0:
                        ax4.text(0.02, 0.02, 'Overfitting Detected!', 
                                transform=ax4.transAxes, verticalalignment='bottom',
                                bbox=dict(boxstyle='round', facecolor='red', alpha=0.8, color='white'))
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Eğitim grafikleri {plot_path} konumuna kaydedildi.")
            
            # Also save as PDF
            pdf_path = os.path.join(output_dir, 'training_curves.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("Matplotlib bulunamadı. Grafik oluşturulamadı.")
            print("Kurulum için: pip install matplotlib")
        except Exception as e:
            print(f"Grafik oluşturma hatası: {e}")
    
    def print_training_summary(self):
        """Print detailed training summary"""
        if not self.training_history:
            print("Eğitim geçmişi bulunamadı.")
            return
        
        print("\n" + "="*60)
        print("EĞİTİM ÖZETİ")
        print("="*60)
        print(f"Model: {self.training_history['model_name']}")
        print(f"Tarih: {self.training_history['timestamp']}")
        print(f"Öğrenme Oranı: {self.training_history['learning_rate']}")
        print(f"Batch Boyutu: {self.training_history['batch_size']}")
        print(f"Epoch Sayısı: {self.training_history['epochs']}")
        print("-"*60)
        print(f"Final Eğitim Kaybı: {self.training_history['final_train_loss']:.6f}")
        if self.training_history['eval_losses']:
            print(f"En İyi Doğrulama Kaybı: {min(self.training_history['eval_losses']):.6f}")
        print(f"Eğitim Süresi: {self.training_history['train_runtime']:.2f} saniye")
        print(f"Saniyede İşlenen Örnek: {self.training_history['train_samples_per_second']:.2f}")
        print("="*60)

if __name__ == "__main__":
    print("T5 Model Eğitimi Başlıyor...")
    print("GPU kullanımı:", "Evet" if torch.cuda.is_available() else "Hayır")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    try:
        # Create model
        model = PhoneT5Model('t5-small')  # t5-base için daha iyi sonuç ama daha yavaş
        model.model.to(device)
        
        # Hyperparameter configuration
        config = {
            'epochs': 15,  # Daha fazla epoch
            'batch_size': 4 if torch.cuda.is_available() else 2,
            'learning_rate': 3e-5,  # Daha düşük LR - daha dikkatli öğrenme
        }
        
        print(f"Eğitim yapılandırması: {config}")
        
        # Train
        train_result = model.train('training_data.txt', **config)
        
        # Print summary
        model.print_training_summary()
        
        print("\nEğitim tamamlandı!")
        print("Dosyalar:")
        print("  - Model: ./phone_t5_model/")
        print("  - Eğitim geçmişi: ./phone_t5_model/training_history.json")
        print("  - Grafikler: ./phone_t5_model/training_curves.png")
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        print("Gerekli paketler: pip install torch transformers datasets matplotlib")
