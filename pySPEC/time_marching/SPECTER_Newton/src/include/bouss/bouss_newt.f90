!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!Newton Solver!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!! Algorithm for solving near recurrences of Rayleigh Benard flow.

!! A solution for a recurrence is such that F(X) = 0, where X = (Omega_0, s_x, T) being Omega_0: velocity and thermal  
!! field at time t_0, s_x the guessed shift for relative periodic orbit, and T the guessed period of the orbit.  

!! F(X) is the difference between the shifted velocity and thermal field at time t+T (Omega_T) and Omega_0 

!! Each Newton step involves solving for s_k the equation J(X_k)*s_k = -F(X_k) (1) and then computing X_k+1 = X_k+s_k.
!! J(X_k) is the Jacobian of F(X)

!!Explain extra 2 equations -> AX = b

!! We solve AX = b by means of GMRES:



! !Save initial field in 1d variable X0
! CALL FlattenFields(X0, vx, vy, vz, th) 

!complex to real
rmp = 1.0_GP/ &
   (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))

DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = vx(k,j,i)*rmp
         C2(k,j,i) = vy(k,j,i)*rmp
         C3(k,j,i) = vz(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 

! !Last slice (containing continuation points)
! IF (kend.gt.kend_ph) THEN
! DO k = kend_ph+1,nz
!    DO j = 1,ny
!       DO i = 1,nx
!          R1(i,j,k) = 0.0_GP
!          R2(i,j,k) = 0.0_GP
!          R3(i,j,k) = 0.0_GP
!       END DO
!    END DO
! END DO
! ENDIF

!save files
CALL io_write(1,odir,'vx_nc1',ext,planio,R1)
CALL io_write(1,odir,'vy_nc1',ext,planio,R2)
CALL io_write(1,odir,'vz_nc1',ext,planio,R3)

!real to complex
CALL fftp3d_real_to_complex(planfc,R1,vx,MPI_COMM_WORLD)
CALL fftp3d_real_to_complex(planfc,R2,vy,MPI_COMM_WORLD)
CALL fftp3d_real_to_complex(planfc,R3,vz,MPI_COMM_WORLD)

DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = th(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)

! !Last slice (containing continuation points)
! IF (kend.gt.kend_ph) THEN
! DO k = kend_ph+1,nz
!    DO j = 1,ny
!       DO i = 1,nx
!          R1(i,j,k) = 0.0_GP
!       END DO
!    END DO
! END DO
! ENDIF

CALL io_write(1,odir,'th_nc1',ext,planio,R1)

CALL fftp3d_real_to_complex(planfc,R1,th,MPI_COMM_WORLD)

!Complex to real
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = vx(k,j,i)*rmp
         C2(k,j,i) = vy(k,j,i)*rmp
         C3(k,j,i) = vz(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 

!save files
CALL io_write(1,odir,'vx_nc2',ext,planio,R1)
CALL io_write(1,odir,'vy_nc2',ext,planio,R2)
CALL io_write(1,odir,'vz_nc2',ext,planio,R3)

DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = th(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)

!save files
CALL io_write(1,odir,'th_nc2',ext,planio,R1)

! pr0 = pr

! !evolve fields dt to compute f(X0)
! DO i = ista,iend
! !$omp parallel do if (iend-ista.lt.nth) private (k)
!     DO j = 1,ny
!         DO k = 1,nz

!         INCLUDE 'include/bouss/bouss_rkstep1.f90'

!         END DO
!     END DO
! END DO

! ! Runge-Kutta step 2
! ! Evolves the system in time

! DO o = ord,1,-1
!     INCLUDE 'include/bouss/bouss_rkstep2.f90'
! END DO

! CALL FlattenFields(X1, vx, vy, vz, th) 

! !Save derivative in variable f_X
! DO i = 1,n_dim_1d
! f_X0(i) = (X1(i)-X0(i))/dt
! ENDDO

! CALL UnflattenFields(vx, vy, vz, th, X0)
! pr = pr0

! !performs evolution of vx, vy, vz, th in time T
! INCLUDE 'include/bouss/bouss_evol_T.f90' 

! !Check that divergence and bc are satisfied
! CALL vdiagnostic(vplanbc,planfc,vx,vy,vz,n_newt+1,1.0_GP)
! CALL sdiagnostic(splanbc,planfc,th,n_newt+1,1.0_GP)

! !Transforms to a 1d variable X_evol
! CALL FlattenFields(X_evol, vx, vy, vz, th)

! !evolve fields dt to compute f(Y)
! DO i = ista,iend
! !$omp parallel do if (iend-ista.lt.nth) private (k)
!     DO j = 1,ny
!         DO k = 1,nz

!         INCLUDE 'include/bouss/bouss_rkstep1.f90'

!         END DO
!     END DO
! END DO

! ! Runge-Kutta step 2
! ! Evolves the system in time

! DO o = ord,1,-1
!     INCLUDE 'include/bouss/bouss_rkstep2.f90'

! END DO

! CALL FlattenFields(X1, vx, vy, vz, th) 

! !Save derivative in variable f_Y
! DO i = 1,n_dim_1d
! f_Y(i) = (X1(i)-X_evol(i))/dt
! ENDDO

! !Traslate in the guessed shifts:
! CALL Translation(Y0, X_evol, 1, sx) 
! CALL Translation(Y0, Y0, 2, sy) 

! CALL Translation(f_Y, f_Y, 1, sx) 
! CALL Translation(f_Y, f_Y, 2, sy)

! DO i = 1,n_dim_1d
!    b(i) = X0(i) - Y0(i) 
! ENDDO

! CALL Norm(b_norm,b)
! IF (myrank.eq.0) THEN
! print '(A,es10.3)', '|b|=',b_norm
! ENDIF

! Res = b/b_norm
! Q(:,1) = Res

! dT_guess= 0.0_GP
! d_sx= 0.0_GP
! d_sy= 0.0_GP
! dX = Res

! DO i=1,3
! Q_aux(i,1) = 0.0_GP
! ENDDO

! e(1) = 1.0_GP
! beta(1) = b_norm


! CALL Norm(aux_norm,X0)

! IF (myrank.eq.0) THEN
! OPEN(1,file='prints/b_error.txt',position='append')
! WRITE(1,FMT='(I2,A,es10.3)') n_newt-1, ',', b_norm
! CLOSE(1)
! OPEN(1,file='prints/solver.txt',position='append')
! WRITE(1,FMT='(I2,A,es10.3,A,es10.3,A,es10.3,A,es10.3)') n_newt, ',', T_guess, ',', sx, ',', sy, ',', aux_norm
! CLOSE(1)
! ENDIF

! CALL UnflattenFields(vx, vy, vz, th, Y0)

! !Save files
! rmp = 1.0_GP/ &
!    (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))

! DO i = ista,iend
! !$omp parallel do if (iend-ista.lt.nth) private (k)
!    DO j = 1,ny
!       DO k = 1,nz
!          C1(k,j,i) = vx(k,j,i)*rmp
!          C2(k,j,i) = vy(k,j,i)*rmp
!          C3(k,j,i) = vz(k,j,i)*rmp
!       END DO
!    END DO
! END DO
! CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
! CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
! CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 
! CALL io_write(1,odir,'vx_T',ext,planio,R1)
! CALL io_write(1,odir,'vy_T',ext,planio,R2)
! CALL io_write(1,odir,'vz_T',ext,planio,R3)


! !$omp parallel do if (iend-ista.ge.nth) private (j,k)
! DO i = ista,iend
! !$omp parallel do if (iend-ista.lt.nth) private (k)
!    DO j = 1,ny
!       DO k = 1,nz
!          C1(k,j,i) = th(k,j,i)*rmp
!       END DO
!    END DO
! END DO
! CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
! CALL io_write(1,odir,'th_T',ext,planio,R1)
! !!!!! End file save


! GMRES: DO n = 1,n_max
   
!    !!! We then compute the LHS of (1) term by term:

!    !Calculate the directional derivative
!    CALL Perturb(X_pert,epsilon,X0,dX )
!    !TODO: check if perturbing with non solenoidal field is troublesome
!    CALL UnflattenFields(vx, vy, vz, th, X_pert)

!    CALL vdiagnostic(vplanbc,planfc,vx,vy,vz,n+1,1.0_GP)
!    CALL sdiagnostic(splanbc,planfc,th,n+1,1.0_GP)

!    !pr field used for boundary conditions not what it should
!    pr = pr0
!    INCLUDE 'include/bouss/bouss_evol_T.f90' 

!    CALL vdiagnostic(vplanbc,planfc,vx,vy,vz,n+1,1.0_GP)
!    CALL sdiagnostic(splanbc,planfc,th,n+1,1.0_GP)

!    !Transforms to a 1d variable X_evol
!    CALL FlattenFields(X_pert_evol, vx, vy, vz, th) 

!    !Calculates the directional derivative term    
!    CALL X_fin_diff(X_partial_dif, X_pert_evol, Y0, sx, sy, epsilon)

!    !Computes terms of translation generator:
!    CALL Shift_term(Y_shift_x, Y0, 1, d_sx) 
!    CALL Shift_term(Y_shift_y, Y0, 2, d_sy) 

!    !!! Now for the rest of the first column of A:

!    !Calculates the projection along the shifted directions and along the direction of flow

!    CALL CalculateProjection(proj_f, proj_x, proj_y, dX, X0, f_X0)

!    CALL Norm(aux_norm, X_partial_dif)

!    ! Can now form r_n = A*r_(n-1)
!    CALL Form_Res(Res, Res_aux, dX, X_partial_dif, f_Y, dT_guess ,Y_shift_x, Y_shift_y, &
!    proj_f, proj_x, proj_y)

!    CALL Arnoldi_step(Res, Res_aux, Q, Q_aux, H, res_norm, n)

!    CALL Update_values(dX, d_sx, d_sy, dT_guess, Res, Res_aux)

!    CALL Givens_rotation(H, cs, sn, n)

!    CALL Update_error(beta, cs, sn, e, b_norm, res_norm, n)


!    IF (myrank.eq.0) THEN
!    OPEN(1,file='prints/error_gmres.txt',position='append')
!    WRITE(1,FMT='(I2,A,I3,A,es10.3)') n_newt, ',', n, ',',e(n+1)
!    CLOSE(1)

!    OPEN(1,file='prints/apply_A.txt',position='append')
!    WRITE(1,FMT='(I2,A,I3,A,es10.3,A,es10.3,A,es10.3,A,es10.3)') n_newt, ',',n, ',', aux_norm, ',',&
!    proj_f, ',' ,proj_x, ',', proj_y
!    CLOSE(1)
!    ENDIF

!    IF (e(n+1)<tol) THEN
!       IF (myrank.eq.0) THEN
!       print *, 'GMRES Convergence'
!       ENDIF
!       n_last = n
!       EXIT GMRES
!    ENDIF

! END DO GMRES


! ! Hookstep: moving the solution to the "trust region"
! ! Delta will delimit our trust region (i.e. |y_sol|<Delta).
! ! In the inner loop (j) we adjust mu_hook s.t. |y_sol|<Delta.
! ! Initially we increment mu_hook in steps of 1, but it can be improved.
! ! In the outer loop (i) once the inner loop is done we check if
! ! the linearized problem is valid, i.e. we check |F(X+dX)|<=|F(X)|.
! ! For this purpose X+dX must be evolved and translated. If
! ! the condition does not hold Delta must be reduced (reduce
! ! the trust region).

! CALL Backpropagation(y_sol, H, beta)
! y_norm = SQRT(SUM(ABS(y_sol)**2))
! Delta = y_norm

! Hookstep: DO n = 1,n_max_hook !TODO: check if a new variable is preferable (n_max_hook) 

!    CALL Hookstep_iter(y_sol, Delta, H, beta)

!    !Compute F(X+dX) to check if |F(X+dX)|<=|F(X)|
!    CALL Update_values2(dX, sx, sy, T_guess,d_sx, d_sy, dT_guess, Q, Q_aux, y_sol)

!    X_pert = X0+dX

!    CALL UnflattenFields(vx, vy, vz, th, X_pert)


!    !!!!!!!Solenoidal Projection!!!!!!!!!
!    ! CALL UnflattenFields(vx, vy, vz, th, dX)

!    CALL vdiagnostic(vplanbc,planfc,vx,vy,vz,-n_newt+1,1.0_GP)
!    CALL sdiagnostic(splanbc,planfc,th,-n_newt+1,1.0_GP)

!    ! Apply boundary conditions and project onto solenoidal space
!     !Option 1: Same as in rkstep2. Added condition in subrout such that o=-1 prevents parameter 
!     ! o from modifying values 
!    ! o = -1
!    ! CALL v_imposebc_and_project(vplanbc,planfc,vx,vy,vz,pr,rki=o,&
!    !          v_zsta=(/ vxzsta, vyzsta/), v_zend=(/ vxzend, vyzend/))

!     !Option 2: Only use the subroutines from sol_projection, without imposing bc. because this uses
!     !pressure field which we do not have for the modified field
!    CALL sol_project(vx,vy,vz,pr,1,0,0)

!    CALL s_imposebc(splanbc,planfc,th)
!    CALL fc_filter(th)

!    ! Hack to prevent weird accumulation in complex domain for th
!    CALL fftp3d_complex_to_real(planfc,th,R1,MPI_COMM_WORLD)
!    R1 = R1/nx/ny/nz
!    CALL fftp3d_real_to_complex(planfc,R1,th,MPI_COMM_WORLD)

!    CALL vdiagnostic(vplanbc,planfc,vx,vy,vz,n_newt+1,1.0_GP)
!    CALL sdiagnostic(splanbc,planfc,th,n_newt+1,1.0_GP)

!    ! CALL FlattenFields(dX, vx, vy, vz, th)
!    CALL FlattenFields(X_pert, vx, vy, vz, th)
!    !!!!!! End Solenoidal Projection

!    ! X_pert = X0+dX

!    ! CALL UnflattenFields(vx, vy, vz, th, X_pert)

!    ! CALL vdiagnostic(vplanbc,planfc,vx,vy,vz,n_newt+1,1.0_GP)
!    ! CALL sdiagnostic(splanbc,planfc,th,n_newt+1,1.0_GP)


!    !!!!!!!!!!!!! F(X+dX)<F(X) !!!!!!!!!!!!!!!!!!
!    !performs evolution of vx, vy, vz, th in time T
!    !TODO: boundary conditions will not match the pressure that has been computed for time T
!    INCLUDE 'include/bouss/bouss_evol_T.f90' 

!    CALL vdiagnostic(vplanbc,planfc,vx,vy,vz,n_newt+1,1.0_GP)
!    CALL sdiagnostic(splanbc,planfc,th,n_newt+1,1.0_GP)

!    !Transforms to a 1d variable X_evol
!    CALL FlattenFields(X_evol, vx, vy, vz, th)

!    !Traslate in the guessed shifts:
!    CALL Translation(Y0, X_evol, 1, sx) 
!    CALL Translation(Y0, Y0, 2, sy) 

!    dX = X_pert - Y0

!    CALL Norm(b_new_norm,dX)

!    IF (myrank.eq.0) THEN
!    OPEN(1,file='prints/hookstep.txt',position='append')
!    WRITE(1,FMT='(I2,A,I2,A,es10.3,A,es10.3,A,es10.3,A,es10.3,A,es10.3)') n_newt, ',', n, ',', &
!    b_new_norm, ',', Delta, ',', T_guess, ',', sx, ',', sy
!    CLOSE(1)  
!    ENDIF

!    IF (b_new_norm.le.b_norm) THEN
!       EXIT
!    ELSE
!       Delta = Delta*0.5_GP
!       T_guess = T_guess - dT_guess
!       sx = sx - d_sx
!       sy = sy - d_sy
!       ! IF (b_new_norm.le.b_norm) THEN
!       ! b_prev_norm = b_new_norm
!       ! ENDIF
!    ENDIF
! END DO Hookstep

! CALL UnflattenFields(vx, vy, vz, th, X_pert)