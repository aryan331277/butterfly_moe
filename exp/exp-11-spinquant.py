def exp11_spinquant_comparison(train_loader, val_loader, vocab_size, device,
                                d_model, num_layers, num_experts, top_k, num_epochs):

    results = {}

    # SpinQuant-style (global rotation baseline)
    spin = SpinQuant_LM(vocab_size, d_model, num_layers).to(device)
    _, ppl_spin = train_model(spin, train_loader, val_loader, device,
                               num_epochs=min(num_epochs, 20), model_name='spincomp_spin',max_steps=MAX_STEPS)
    spin_rot_mb = (d_model * d_model * 4) / (1024 * 1024)
    spin_w_mb   = (d_model * d_model * 4 * 1.58) / (8 * 1024 * 1024)
    mem_spin = spin_rot_mb + spin_w_mb
    results['SpinQuant-style\n(global rot.)'] = {'ppl': ppl_spin, 'mem_mb': mem_spin}
    del spin
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print("\n" + "-"*60)
    print(f"{'Method':<35} {'Val PPL':<12} {'Memory MB':<12}")
    print("-"*60)
    for name, r in results.items():
        clean = name.replace('\n', ' ')
        print(f"{clean:<35} {r['ppl']:<12.2f} {r['mem_mb']:<12.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_map = {
        'SpinQuant-style\n(global rot.)':  '#FF9800',
    }
    for name, r in results.items():
        axes[0].scatter(r['mem_mb'], r['ppl'], s=300,
                        c=colors_map[name], edgecolors='black', zorder=5, label=name)
    axes[0].set_xlabel('Memory (MB)'); axes[0].set_ylabel('Validation Perplexity')
    axes[0].set_title('PPL vs Memory: SpinQuant-style\n(lower-left = better)')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    names_clean = list(results.keys())
    ppls_list   = [r['ppl'] for r in results.values()]
    bar_colors  = [colors_map[n] for n in results.keys()]
    axes[1].bar(range(len(results)), ppls_list, color=bar_colors, edgecolor='black')
    axes[1].set_xticks(range(len(results)))
    axes[1].set_xticklabels(names_clean, fontsize=9)
    axes[1].set_ylabel('Validation Perplexity')
    axes[1].set_title('Perplexity: SpinQuant-style')
    axes[1].grid(alpha=0.3, axis='y')
    for i, v in enumerate(ppls_list):
        axes[1].text(i, v + 0.3, f'{v:.2f}', ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('exp11_spinquant_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    return results
