import matplotlib.pyplot as plt

from butterflymoe.models.language_models import ButterflyMoE_LM_NBit
from butterflymoe.training.trainer import train_model, MAX_STEPS


def exp10_accuracy_vs_bitwidth(train_loader, val_loader, vocab_size, device,
                                d_model, num_layers, num_experts, top_k, num_epochs):

    configs = [
        {'n_bits': 2,    'label': '2-bit',         'bpw': 2.0},
        {'n_bits': 1,    'label': '1-bit',         'bpw': 1.0}
    ]

    results = []
    for cfg in configs:
        print(f"\n  Training with {cfg['label']} substrate...")
        model = ButterflyMoE_LM_NBit(vocab_size, d_model, num_layers,
                                      num_experts, top_k, n_bits=cfg['n_bits']).to(device)
        _, best_ppl = train_model(model, train_loader, val_loader, device,
                                   num_epochs=min(num_epochs, 15),
                                   model_name=f'bitwidth_{cfg["label"].replace(".", "_")}',max_steps=MAX_STEPS)
        substrate_mb = (d_model * d_model * 4 * cfg['bpw']) / (8 * 1024 * 1024)
        results.append({
            'label': cfg['label'], 'n_bits': cfg['n_bits'],
            'ppl': best_ppl, 'substrate_mb': substrate_mb,
        })
        print(f"  {cfg['label']}: PPL={best_ppl:.2f}, Substrate={substrate_mb:.3f}MB")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n" + "-"*55)
    print(f"{'Bitwidth':<20} {'Val PPL':<15} {'Substrate MB':<15}")
    print("-"*55)
    for r in results:
        print(f"{r['label']:<20} {r['ppl']:<15.2f} {r['substrate_mb']:<15.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [r['label'] for r in results]
    ppls   = [r['ppl'] for r in results]
    mems   = [r['substrate_mb'] for r in results]

    bar_colors = ['#FF5722' if r['label'] != '1.58-bit (ours)' else '#2196F3' for r in results]

    axes[0].bar(labels, ppls, color=bar_colors, edgecolor='black')
    axes[0].set_ylabel('Validation Perplexity (lower=better)')
    axes[0].set_title('Accuracy vs Substrate Bitwidth')
    axes[0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(ppls):
        axes[0].text(i, v + 0.3, f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)

    axes[1].scatter(mems, ppls, s=200, c=bar_colors, edgecolors='black', zorder=5)
    for i, r in enumerate(results):
        axes[1].annotate(r['label'], (mems[i], ppls[i]),
                         textcoords='offset points', xytext=(5, 5), fontsize=9)
    axes[1].set_xlabel('Substrate Memory (MB)')
    axes[1].set_ylabel('Validation Perplexity')
    axes[1].set_title('Accuracy vs Memory: Bitwidth Pareto\n(lower-left = better)')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Substrate Bitwidth Ablation — justifying 1.58-bit choice',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('exp10_accuracy_vs_bitwidth.png', dpi=150, bbox_inches='tight')
    plt.close()
    return results
