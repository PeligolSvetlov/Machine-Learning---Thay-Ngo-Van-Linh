icks(rotation=30, ha='right')

plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, v in enumerate(area_stats):
    plt.text(i, v + 100, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
plt.close('all')