export function evaluate(ranking: string[], relevance: Record<string, number>, k = 10) {
  const relevant = Object.values(relevance).filter(value => value > 0).length;
  let dcg = 0, hits = 0, reciprocalRank = 0;
  ranking.slice(0, k).forEach((id, rank) => {
    const grade = relevance[id] ?? 0;
    // BEIR uses trec_eval ndcg_cut: gain is the qrel grade, not 2^grade - 1.
    dcg += Math.max(0, grade) / Math.log2(rank + 2);
    if (grade > 0) { hits++; if (!reciprocalRank) reciprocalRank = 1 / (rank + 1); }
  });
  const ideal = Object.values(relevance).filter(x => x > 0).sort((a,b) => b-a).slice(0,k)
    .reduce((sum, grade, rank) => sum + grade / Math.log2(rank + 2), 0);
  return { ndcg10: ideal ? dcg / ideal : 0, recall10: relevant ? hits / relevant : 0, mrr10: reciprocalRank };
}

/** Paired bootstrap over queries, fixed PRNG seed. No test-set parameter tuning. */
export function pairedBootstrap(differences: number[], samples = 2000): [number, number] {
  if (!differences.length) return [0,0];
  let state = 42;
  const random = () => ((state = (Math.imul(1664525, state) + 1013904223) >>> 0) / 2 ** 32);
  const means = Array.from({length: samples}, () => {
    let sum = 0;
    for (let i = 0; i < differences.length; i++) sum += differences[Math.floor(random() * differences.length)];
    return sum / differences.length;
  }).sort((a,b) => a-b);
  return [means[Math.floor(samples * 0.025)], means[Math.floor(samples * 0.975)]];
}
