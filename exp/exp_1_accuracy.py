def exp1_main_accuracy_table(train_loader, val_loader, vocab_size, device,
                              d_model, num_layers, num_experts, top_k, num_epochs):

    results = {}

    # --- ButterflyMoE ---
    print("\nTraining ButterflyMoE...")
    bmoe = ButterflyMoE_LM(vocab_size, d_model, num_layers, num_experts, top_k).to(device)
    hist_bmoe, best_ppl_bmoe = train_model(bmoe, train_loader, val_loader, device,
                                            #num_epochs=num_epochs, model_name='butterflymoe',max_steps=MAX_STEPS)
    _, final_ppl_bmoe = evaluate(bmoe, val_loader, device)
    results['ButterflyMoE'] = {'ppl': final_ppl_bmoe, 'history': hist_bmoe}

    # --- Standard MoE ---
    print("\nTraining Standard MoE...")
    smoe = StandardMoE_LM(vocab_size, d_model, num_layers, num_experts, top_k).to(device)
    hist_smoe, best_ppl_smoe = train_model(smoe, train_loader, val_loader, device,
                                            #num_epochs=num_epochs, model_name='standardmoe',max_steps=MAX_STEPS)
    _, final_ppl_smoe = evaluate(smoe, val_loader, device)
    results['StandardMoE'] = {'ppl': final_ppl_smoe, 'history': hist_smoe}

    # --- Dense FFN ---
    print("\nTraining Dense FFN...")
    dense = Dense_LM(vocab_size, d_model, num_layers).to(device)
    hist_dense, best_ppl_dense = train_model(dense, train_loader, val_loader, device,
                                              num_epochs=num_epochs, model_name='dense',max_steps=MAX_STEPS)
    _, final_ppl_dense = evaluate(dense, val_loader, device)
    results['Dense'] = {'ppl': final_ppl_dense, 'history': hist_dense}

    # Memory
    mem_bmoe = calculate_memory_mb(d_model, d_model * 4, num_experts, 'butterfly')
    mem_smoe = calculate_memory_mb(d_model, d_model * 4, num_experts, 'standard')
    mem_dense = (d_model * d_model * 4 * 3) / (1024 * 1024)  # approx 3 dense matrices

    print("\n" + "-"*70)
    print(f"{'Method':<20} {'Val PPL':<15} {'Memory (MB)':<15} {'Compression':<15}")
    print("-"*70)
    print(f"{'Dense FFN':<20} {final_ppl_dense:<15.2f} {mem_dense:<15.3f} {'1.0x':<15}")
    print(f"{'Standard MoE':<20} {final_ppl_smoe:<15.2f} {mem_smoe:<15.3f} {'1.0x (ref)':<15}")
    print(f"{'ButterflyMoE':<20} {final_ppl_bmoe:<15.2f} {mem_bmoe:<15.3f} {mem_smoe/mem_bmoe:<15.1f}x")
    print("-"*70)

    # Plot convergence curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, num_epochs + 1)

    for name, color, hist in [
        ('ButterflyMoE', '#2196F3', hist_bmoe),
        ('Standard MoE', '#FF5722', hist_smoe),
        ('Dense FFN',    '#4CAF50', hist_dense),
    ]:
        axes[0].plot(epochs, hist['train_loss'], label=name, color=color, linewidth=2)
        axes[1].plot(epochs, hist['val_ppl'],    label=name, color=color, linewidth=2)

    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss Convergence'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation Perplexity (lower=better)')
    axes[1].set_title('Validation Perplexity'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('exp1_accuracy_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results, bmoe, smoe, dense
