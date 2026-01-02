from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit, 
                             QDialogButtonBox, QFormLayout, QSpinBox)

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(300, 200)
        
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.camera_id_input = QSpinBox()
        self.camera_id_input.setValue(0)
        form_layout.addRow("Camera ID:", self.camera_id_input)
        
        self.threshold_input = QSpinBox()
        self.threshold_input.setRange(5, 300)
        self.threshold_input.setValue(30)
        form_layout.addRow("Alert Threshold (s):", self.threshold_input)
        
        layout.addLayout(form_layout)
        
        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_settings(self):
        return {
            'camera_id': self.camera_id_input.value(),
            'threshold': self.threshold_input.value()
        }
