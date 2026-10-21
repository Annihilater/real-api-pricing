"""Recompute the anonymous Magpie evidence without private logs or SciPy."""
import json
import math
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'data/research/magpie-local-tokyo-round1-2026-10-21.json'


def close(actual, expected, label, absolute=1e-8):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=absolute), (label, actual, expected)


def moment(text):
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    assert value.tzinfo is not None, text
    return value


def main():
    text = PATH.read_text(encoding='utf-8')
    data = json.loads(text)
    assert data['adoption_status'].startswith('Research-only')
    assert not re.search(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}', text), 'Email in public evidence'
    assert not re.search(r'(?i)(?:(?<![a-z])[a-z]:[\\/]|/root/|/opt/|bearer\s+\S+|gh[pousr]_[a-z0-9]+)', text)
    assert not re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text), 'IP address in public evidence'
    for forbidden in ('response_id', 'session_account', 'providerAccount', 'callerKeyId', 'request_id'):
        assert f'"{forbidden}"' not in text, forbidden
    collection = data['collection']
    assert sum(collection['input_gateway_rows'].values()) - collection['mirrored_forwarding_rows_removed'] == collection['merged_gateway_rows']
    assert collection['gateway_rows_at_or_before_cutoff'] <= collection['merged_gateway_rows']
    for host in data['source_files'].values():
        for source in host.values():
            assert re.fullmatch('[0-9a-f]{64}', source['sha256'])
            assert source['bytes'] > 0

    convention = data['reference_repository']['conventions']
    assert re.fullmatch('[0-9a-f]{40}', data['reference_repository']['revision'])
    ids = set()
    for m in data['measurements']:
        assert m['id'] not in ids
        ids.add(m['id'])
        assert m['evidence_grade'] in ('medium', 'high') and m['adopted'] is False
        assert m['scope'] == 'conditional_account_cycle_capacity'
        assert moment(m['period_end']) <= moment(data['observation_cutoff'])
        selected = [s for s in m['samples'] if s['used_for_this_model']]
        assert len(selected) >= 2
        numerators, deltas, calls = [], [], 0
        blend = m['reference_blend_usd_per_MTok']
        if m['normalized']:
            key = {'standard': 'standardTokenMix', 'anthropic': 'anthropicTokenMix',
                   'low_cache': 'lowCacheTokenMix'}[m['workload']]
            mix = convention[key]
            rates = m['reference_rates_usd_per_MTok']
            fresh_rate = rates['cache_write_5m'] if m['workload'] == 'anthropic' else rates['input']
            fresh_share = mix.get('input', mix.get('cacheWrite', 0))
            close(blend, mix['cache']*rates['cache_read'] + fresh_share*fresh_rate + mix['output']*rates['output'], m['id'])
        else:
            assert blend is None and m['reference_rates_usd_per_MTok'] is None
            assert m['model'] == 'minimax-m3.1-flash-preview', 'Unapproved raw exception'
        for s in selected:
            assert s['status'] == 'eligible' and s['pct_consumed'] > 0
            active = {name:v for name,v in s['models'].items() if v['tokens']['total'] > 0}
            assert set(active) == {m['telemetry_model']}, 'Mixed-pool capacity admitted'
            part = active[m['telemetry_model']]
            t = part['tokens']
            assert part['full_token_total_known']
            assert all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in t.values())
            assert t['cache_write_1h'] <= t['cache_write']
            assert t['total'] == sum(t[f] for f in ('in','out','cache_read','cache_write'))
            if m['normalized']:
                worth = (t['in']*rates['input'] + t['out']*rates['output'] +
                         t['cache_read']*rates['cache_read'] +
                         (t['cache_write']-t['cache_write_1h'])*rates['cache_write_5m'] +
                         t['cache_write_1h']*rates['cache_write_1h']) / 1e6
                close(part['reference_worth_usd'], worth, m['id'])
                numerators.append(worth/blend*1e6)
            else:
                numerators.append(t['total'])
            deltas.append(s['pct_consumed'])
            calls += part['calls']
        pp = sum(deltas)
        capacity = sum(numerators)/(pp/100)
        close(m['delta_pp'], pp, m['id'])
        assert m['calls'] == calls
        close(m['weekly_tokens'], capacity, m['id'], absolute=0.51)
        close(m['monthly_tokens'], m['weekly_tokens']*m['weeks_per_comparison_month'], m['id'])
        assert m['weeks_per_comparison_month'] == (5 if m['provider']=='kimi-code-cn' else convention['monthWeeks'])
        fee = m['reference_monthly_fee']['value']
        close(m['saturated_reference_price_native_per_MTok'], fee/(m['monthly_tokens']/1e6), m['id'])

        a = m['confidence_assessment']
        effective = pp**2 / sum(d*d for d in deltas)
        cv = math.sqrt(sum(d*(n/(d/100)-capacity)**2 for n,d in zip(numerators,deltas))/pp)/capacity
        loo = max(abs((sum(numerators)-n)/((pp-d)/100)/capacity-1) for n,d in zip(numerators,deltas))
        runs = 1 + sum(x['period_end'] != y['period_start'] for x,y in zip(selected,selected[1:]))
        close(a['effective_segments'], effective, m['id'])
        close(a['weighted_cv'], cv, m['id'])
        close(a['leave_one_out_max_relative_change'], loo, m['id'])
        close(a['rounding_delta_relative_error'], runs/pp, m['id'])
        assert a['segments'] == len(selected) and a['model_calls'] == calls
        assert a['grade'] == m['evidence_grade'] == 'medium', 'Audit completeness caps grades'
        rule = data['qualification_policy']['medium']
        assert pp >= rule['delta_pp'] and calls >= rule['model_calls']
        assert effective >= rule['effective_segments'] and cv <= rule['weighted_cv']
        assert loo <= rule['leave_one_out_change'] and runs/pp <= rule['rounding_ratio']
        assert a['known_portion_unattributed_ratio'] <= data['qualification_policy']['direct_ratio_low_cap']
        assert m['rounding_sensitivity_is_confidence_interval'] is False
        low, high = m['rounding_sensitivity_weekly_tokens']
        close(low, sum(numerators)/((pp+runs)/100), m['id'], absolute=0.51)
        close(high, sum(numerators)/((pp-runs)/100), m['id'], absolute=0.51)
        if m['account_ref'].startswith('devin_'):
            assert m['plan_identity']['basis'] == 'contributor_confirmation'
        if m['account_ref'] == 'devin_max_special_no_daily_cap':
            assert 'special' in m['plan'] and 'no daily cap' in m['plan_identity']['note']
    assert len(ids) == 7
    for account in data['inventory']:
        assert set(account['qualified_measurement_ids']) <= ids
    for excluded in data['excluded_observations']:
        assert excluded['id'] not in ids
        assert 'weekly_tokens' not in excluded, 'Excluded capacity must not be published as usable'
    goat, = data['official_facts']
    assert goat['evidence_grade']=='high' and goat['type']=='official_credit_terms'
    assert goat['usage_limits_table_monthly_credits']==70 and goat['plan_page_headline_credits']==80
    assert goat['five_hour_usage_credits']==14 and goat['weekly_usage_credits']==35
    obs=goat['local_cached_observation']
    close(round(obs['monthly_used_percent']/100*obs['monthly_display_cap'],2),obs['monthly_display_used'],'GOAT display',absolute=0.005)
    assert all(url.startswith('https://commandcode.ai/') for url in goat['urls'])
    print(f'PASS: {len(ids)} medium measurements, one high-grade official observation, '
          f'{len(data["excluded_observations"])} exclusions, {len(data["inventory"])} anonymous accounts')


if __name__ == '__main__':
    main()
