"""Ad-hoc checks of the controllable-turn rule in embody.py (turn_min, turn_margin, turn_st). Run: python lab/test_turn_rule.py (from lab/)."""
import sys; sys.path.insert(0,'/home/kuans/PROJ/microbots/phase3/npa'); sys.path.insert(0,'/home/kuans/PROJ/microbots/phase3/npa/lab')
import jax, jax.numpy as jnp, heading as hd, embody
class B: mem_dims=0
def run(**kw):
    e=embody.Embodied(B(),mode="phys",axis_lock=True,axis_turn=True,turn_shift=0.0284,max_step=0.043,step_len_noise=0.0,axis_drift=0.0,**kw)
    th=hd.headings(4,0); a=jnp.stack([jnp.cos(th),jnp.sin(th)],-1); u=jnp.stack([-jnp.sin(th),jnp.cos(th)],-1)
    # bots 0..3 on the 'u' line (L=0): 0 noise-size other-line, 1 deliberate other-line, 2 along own line, 3 mostly other but 1.5x only
    d=jnp.stack([1e-4*a[0], 0.03*a[1], 0.03*u[2], 0.03*a[3]+0.02*u[3]],0)
    L=jnp.zeros((4,1))
    d2,L2=e._stride(d,jax.random.PRNGKey(0),L)
    return [round(float(v),0) for v in L2[:,0]], [round(float(jnp.linalg.norm(x))*1000,1) for x in d2]
print("default      ",run())
print("min .01 m 2  ",run(turn_min=0.01,turn_margin=2.0))

def grad_check(**kw):
    e=embody.Embodied(B(),mode="phys",axis_lock=True,axis_turn=True,turn_shift=0.0284,max_step=0.043,step_len_noise=0.0,axis_drift=0.0,turn_min=0.004,turn_margin=1.3,**kw)
    th=hd.headings(2,0); a=jnp.stack([jnp.cos(th),jnp.sin(th)],-1); u=jnp.stack([-jnp.sin(th),jnp.cos(th)],-1)
    L=jnp.zeros((2,1)); key=jax.random.PRNGKey(0)
    d0=jnp.stack([0.01*u[0], 0.03*a[1]],0)     # bot 0 strides on its line, bot 1 asks for the other line (turns)
    f=lambda d: e._stride(d,key,L)[0]
    out=f(d0)
    # gradient of the executed move of bot 0 wrt its proposal component along the OTHER line (a)
    g=jax.grad(lambda t: jnp.sum(f(d0.at[0].add(t*a[0]))[0]*a[0]))(0.0)
    g1=jax.grad(lambda t: jnp.sum(f(d0.at[1].add(t*a[1]))[1]*a[1]))(0.0)
    return out, float(g), float(g1)
o0,g0,h0=grad_check(); o1,g1,h1=grad_check(turn_st=1.0)
print("forward equal:", bool(jnp.allclose(o0,o1)), "| grad other-line, striding bot: off", g0, "on", g1, "| turning bot: off", h0, "on", h1)
