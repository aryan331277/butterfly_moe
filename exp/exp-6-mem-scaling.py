def exp6_memory_scaling(d_model=512, d_ff=512):


    expert_counts = [2, 4, 8, 16, 32, 64, 128, 256]

    standard_mem = [calculate_memory_mb(d_model, d_ff, n, 'standard') for n in expert_counts]
    butterfly_mem= [calculate_memory_mb(d_model, d_ff, n, 'butterfly') for n in expert_counts]
    qmoe_mem     = [calculate_memory_mb(d_model, d_ff, n, 'qmoe') for n in expert_counts]
    moqe_mem     = [calculate_memory_mb(d_model, d_ff, n, 'moqe') for n in expert_counts]

    print(f"\n{'N':>6} {'Standard':>12} {'QMoE(est)':>12} {'MoQE(est)':>12} "
          f"{'ButterflyMoE':>14} {'Compression':>12}")
    print("-"*70)
    for i, n in enumerate(expert_counts):
        comp = standard_mem[i] / butterfly_mem[i]
        print(f"{n:>6} {standard_mem[i]:>12.2f} {qmoe_mem[i]:>12.2f} "
              f"{moqe_mem[i]:>12.2f} {butterfly_mem[i]:>14.3f} {comp:>12.1f}x")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, mem, color, ls in [
        ('Standard MoE', standard_mem, '#FF5722', '-'),
        ('QMoE (est.)',  qmoe_mem,     '#FF9800', '--'),
        ('MoQE 2-bit (est.)', moqe_mem,'#FFC107', '--'),
        ('ButterflyMoE',butterfly_mem, '#2196F3', '-'),
    ]:
        axes[0].plot(expert_counts, mem, 'o-', label=label, linewidth=2,
                     color=color, linestyle=ls)

    axes[0].set_xlabel('Number of Experts'); axes[0].set_ylabel('Memory (MB)')
    axes[0].set_title('Memory Scaling: All Methods')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_yscale('log'); axes[0].set_xscale('log', base=2)

    compression = [standard_mem[i] / butterfly_mem[i] for i in range(len(expert_counts))]
    axes[1].plot(expert_counts, compression, 'o-', color='#2196F3', linewidth=2, markersize=10)
    axes[1].set_xlabel('Number of Experts')
    axes[1].set_ylabel('Compression Ratio vs Standard MoE')
    axes[1].set_title('ButterflyMoE Compression Ratio\n(improves with expert count)')
    axes[1].grid(alpha=0.3); axes[1].set_xscale('log', base=2)
    for i, (n, c) in enumerate(zip(expert_counts, compression)):
        if i % 2 == 0:
            axes[1].annotate(f'{c:.0f}x', (n, c), textcoords='offset points',
                             xytext=(5, 5), fontsize=9)

    plt.tight_layout()
    plt.savefig('exp6_memory_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: exp6_memory_scaling.png")

    return expert_counts, butterfly_mem, standard_mem
