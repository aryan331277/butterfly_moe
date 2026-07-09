def exp12_monarch_comparison(train_loader, val_loader, vocab_size, device,
                              d_model, num_layers, num_experts, top_k, num_epochs):


    results = {}

    print("\n  Training MonarchMoE...")
    monarch = MonarchMoE_LM(vocab_size, d_model, num_layers, num_experts, top_k).to(device)
    _, ppl_monarch = train_model(monarch, train_loader, val_loader, device,
                                  num_epochs=min(num_epochs, 20), model_name='monarch_monarch',max_steps=MAX_STEPS)
    tput_monarch, lat_monarch = measure_throughput(monarch, 16, 64, vocab_size, device, 50)
    block = max(2, int(math.sqrt(d_model)))
    monarch_params_per_expert = 2 * (d_model // block) * block * block
    mem_monarch = (
        calculate_memory_mb(d_model, d_model * 4, 1, 'butterfly') +
        num_experts * monarch_params_per_expert * 4 / (1024 * 1024)
    )
    results['MonarchMoE'] = {'ppl': ppl_monarch, 'throughput': tput_monarch,
                              'latency': lat_monarch, 'mem_mb': mem_monarch}
    del monarch
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print("\n" + "-"*65)
    print(f"{'Method':<18} {'Val PPL':<12} {'Throughput':<15} {'Latency(ms)':<14} {'Mem(MB)':<10}")
    print("-"*65)
    for name, r in results.items():
        print(f"{name:<18} {r['ppl']:<12.2f} {r['throughput']:<15.0f} "
              f"{r['latency']:<14.2f} {r['mem_mb']:<10.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names  = list(results.keys())
    ppls   = [r['ppl']        for r in results.values()]
    tputs  = [r['throughput'] for r in results.values()]
    mems   = [r['mem_mb']     for r in results.values()]
    colors = ['#FF9800']

    axes[0].bar(names, ppls,  color=colors, edgecolor='black')
    axes[0].set_ylabel('Validation Perplexity'); axes[0].set_title('Accuracy')
    axes[0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(ppls):
        axes[0].text(i, v + 0.3, f'{v:.2f}', ha='center', fontweight='bold')

    axes[1].bar(names, tputs, color=colors, edgecolor='black')
    axes[1].set_ylabel('Throughput (tok/s)'); axes[1].set_title('Throughput')
    axes[1].grid(alpha=0.3, axis='y')

    axes[2].bar(names, mems,  color=colors, edgecolor='black')
    axes[2].set_ylabel('Memory (MB)'); axes[2].set_title('Memory Footprint')
    axes[2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('exp12_monarch_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    return results
