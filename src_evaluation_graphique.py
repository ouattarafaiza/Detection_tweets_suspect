# src_evaluation_graphique.py
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

print("=== Partie 6 : Évaluation Graphique (Matrice de confusion & ROC) ===")

# 1. Chargement des modèles entraînés et des données de test isolées
modeles = joblib.load('modeles_entraines.pkl')
X_test, y_test = joblib.load('donnees_test.pkl')

# On sélectionne notre modèle champion : Random Forest
modele_champion = modeles['Random_Forest']

# 2. Prédictions
y_pred = modele_champion.predict(X_test)
y_prob = modele_champion.predict_proba(X_test)[:, 1] # Probabilités pour la courbe ROC

# 3. Génération et sauvegarde de la Matrice de Confusion
print("Génération de la matrice de confusion...")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Suspect (0)', 'Suspect (1)'])

plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Matrice de Confusion - Random Forest")
plt.savefig('matrice_confusion.png', bbox_inches='tight')
plt.close()
print("Image 'matrice_confusion.png' sauvegardée.")

# 4. Génération et sauvegarde de la Courbe ROC & AUC
print("Génération de la courbe ROC...")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.title('Courbe ROC - Random Forest')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('courbe_roc.png', bbox_inches='tight')
plt.close()
print(f"Image 'courbe_roc.png' sauvegardée avec un score AUC de {roc_auc:.4f}.")
print("Étape d'évaluation graphique terminée avec succès !")