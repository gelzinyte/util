import click

@click.command('mem')
@click.argument('my_job_id')
@click.option('--period', '-p', type=click.INT, default=60, help='how often to check')
@click.option('--max_time', type=click.INT, default=1e5, help='How long to keep checking for, s')
@click.option('--out_fname_prefix', default='mem_usage', help='output\'s prefix')
@click.option('--womble', 'cluster', flag_value='womble', default=True)
@click.option('--young', 'cluster', flag_value='young')
def memory(my_job_id, period=10, max_time=100000,
              out_fname_prefix='mem_usage', cluster='womble'):
    from util import mem_tracker
    out_fname = f'{out_fname_prefix}_{my_job_id}.txt'
    mem_tracker.track_mem(my_job_id=my_job_id, period=period, max_time=max_time,
                     out_fname=out_fname, cluster=cluster)