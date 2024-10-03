 def loss(self, step):
        u_loss = np.sum(np.array([(np.load(f'{self.pm.data_path}/uu_{tstep:04}.npy') - np.load(f'{self.pm.field_path}/uu_{tstep:04}.npy'))**2 for tstep in range(int(self.pm.T/self.pm.dt))]))
        h_loss = np.sum(np.array([(np.load(f'{self.pm.data_path}/hh_{tstep:04}.npy') - np.load(f'{self.pm.field_path}/hh_{tstep:04}.npy'))**2 for tstep in range(int(self.pm.T/self.pm.dt))]))

        loss = [f'{self.pm.dt*step:.4e}', f'{u_loss:.6e}' , f'{h_loss:.6e}']
        with open(f'{self.pm.out_path}/loss.dat', 'a') as output:
            print(*loss, file=output)

        hb_val = np.sum((np.load(f'{self.pm.data_path}/hb.npy') - np.load(f'{self.pm.field_path}/hb.npy'))**2)

        loss = [f'{self.pm.dt*step:.4e}', f'{hb_val:.6e}']
        with open(f'{self.pm.out_path}/hb_val.dat', 'a') as output:
            print(*loss, file=output)
