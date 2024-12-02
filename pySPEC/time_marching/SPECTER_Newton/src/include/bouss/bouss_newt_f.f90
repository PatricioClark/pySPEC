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

IF (myrank.eq.0) THEN
OPEN(1,file='error_gmres.txt',position='append')
WRITE(1,FMT='(A, I3)') 'Solver iteration: ', n_newt
CLOSE(1)
OPEN(1,file='debug_gmres.txt',position='append')
WRITE(1,FMT='(A, I3)') 'Solver iteration: ', n_newt
CLOSE(1)
ENDIF

IF (myrank.eq.0) THEN
print '(A,es13.6)', 'T_guess= ', T_guess
print '(A,es13.6)', 'sx= ', sx
print '(A,es13.6)', 'sy= ',sy
ENDIF


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!! Time Deriv check
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! rmp = 1.0_GP/ &
!    (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))

! DO i = ista,iend
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
! CALL io_write(1,odir,'vx',ext,planio,R1)
! CALL io_write(1,odir,'vy',ext,planio,R2)
! CALL io_write(1,odir,'vz',ext,planio,R3)
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
! CALL io_write(1,odir,'th',ext,planio,R1)


! !Save initial field in 1d variable X0
! CALL ThreeTo1D(X0, vx, vy, vz, th) 

! !TODO: check if making a copy of original p makes a difference
! pr0 = pr


! !performs evolution of vx, vy, vz, th in time T
! !INCLUDE 'include/bouss/bouss_evol_T.f90' 

! !CALL ThreeTo1D(X_evol, vx, vy, vz, th) 

! X0 = X1
! !Evolve dt 
! !TODO1: check if making dt smaller for this is better, if possible
! !TODO2: check if using 3 points finite difference is better
!    !TODO: Copy is made just before entering bouss_newt but i leave it for now
!    !$omp parallel do if (iend-ista.ge.nth) private (j,k)
! DO  n = 1, 10
!    DO i = ista,iend
!    !$omp parallel do if (iend-ista.lt.nth) private (k)
!       DO j = 1,ny
!          DO k = 1,nz

!             INCLUDE 'include/bouss/bouss_rkstep1.f90'

!          END DO
!       END DO
!    END DO

!    ! Runge-Kutta step 2
!    ! Evolves the system in time

!    DO o = ord,1,-1
!       INCLUDE 'include/bouss/bouss_rkstep2.f90'
   
!    END DO

!    CALL ThreeTo1D(X2, vx, vy, vz, th) 

!    !Save derivative in variable f_X
!    CALL Deriv_X(f_X0, X2, X1, dt)

!    CALL Norm(aux_norm,f_X0)
!    IF (myrank.eq.0) THEN
!    OPEN(1,file='error_gmres.txt',position='append')
!    WRITE (1,'(I2,A,es10.3)') n,'. Fin diff: |f_X0| = ',aux_norm
!    CLOSE(1)
!    ENDIF

!    CALL OneTo3D(vx,vy,vz,th,f_X0)
!    WRITE(ext, fmtext) tind

!    rmp = 1.0_GP/ &
!       (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))

!    DO i = ista,iend
!       DO j = 1,ny
!          DO k = 1,nz
!             C1(k,j,i) = vx(k,j,i)*rmp
!             C2(k,j,i) = vy(k,j,i)*rmp
!             C3(k,j,i) = vz(k,j,i)*rmp
!          END DO
!       END DO
!    END DO
!    CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
!    CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
!    CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 
!    CALL io_write(1,odir,'f_vx',ext,planio,R1)
!    CALL io_write(1,odir,'f_vy',ext,planio,R2)
!    CALL io_write(1,odir,'f_vz',ext,planio,R3)
!    !$omp parallel do if (iend-ista.ge.nth) private (j,k)
!    DO i = ista,iend
!    !$omp parallel do if (iend-ista.lt.nth) private (k)
!       DO j = 1,ny
!          DO k = 1,nz
!             C1(k,j,i) = th(k,j,i)*rmp
!          END DO
!       END DO
!    END DO
!    CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
!    CALL io_write(1,odir,'f_th',ext,planio,R1)

!    CALL OneTo3D(vx,vy,vz,th,X2)
!    X1 = X2
!    tind = tind+1

! ENDDO



! !Save derivative in variable f_X
! CALL Deriv_X(f_X0, X2,X0, dt*10.0_GP)

! CALL Norm(aux_norm,f_X0)
! IF (myrank.eq.0) THEN
! print '(A,es10.3)', 'Fin diff 10 steps: |f_X0| = ',aux_norm
! ENDIF


!!! Check if the magnitude of f(X0) is very different from the rest
! DO i = 1,260

!    tind = tind+1
!    WRITE(ext, fmtext) tind

!    CALL io_read(1,idir,'vx',ext,planio,R1)
!    CALL io_read(1,idir,'vy',ext,planio,R2)
!    CALL io_read(1,idir,'vz',ext,planio,R3)
!    CALL fftp3d_real_to_complex(planfc,R1,vx,MPI_COMM_WORLD)
!    CALL fftp3d_real_to_complex(planfc,R2,vy,MPI_COMM_WORLD)
!    CALL fftp3d_real_to_complex(planfc,R3,vz,MPI_COMM_WORLD)
!    CALL io_read(1,idir,'th',ext,planio,R1)
!    CALL fftp3d_real_to_complex(planfc,R1,th,MPI_COMM_WORLD)
   
!    CALL ThreeTo1D(X1, vx, vy, vz, th) 

!    !Save derivative in variable f_X
!    CALL Deriv_X(f_X0, X1, X0, dt*50.0_GP)

!    CALL Norm(aux_norm,f_X0)
!    IF (myrank.eq.0) THEN
!    OPEN(1,file='error_gmres.txt',position='append')
!    WRITE (1,'(A,A,es10.3)') ext,'. Fin diff: |f_X0| = ',aux_norm
!    CLOSE(1)
!    ENDIF

!    X0 = X1

! ENDDO







! !!!! End check for f(X0)

! !Compute time derivative of X0 using finite differences f(X0) = (X(t+dt)-X(t))/dt 

! !Evolve dt 
! !TODO1: check if making dt smaller for this is better, if possible
! !TODO2: check if using 3 points finite difference is better
!    !TODO: Copy is made just before entering bouss_newt but i leave it for now
!    !$omp parallel do if (iend-ista.ge.nth) private (j,k)
!    DO i = ista,iend
!    !$omp parallel do if (iend-ista.lt.nth) private (k)
!       DO j = 1,ny
!          DO k = 1,nz

!             INCLUDE 'include/bouss/bouss_rkstep1.f90'

!          END DO
!       END DO
!    END DO

!    ! Runge-Kutta step 2
!    ! Evolves the system in time

!    DO o = ord,1,-1
!       INCLUDE 'include/bouss/bouss_rkstep2.f90'
   
!    END DO


! CALL ThreeTo1D(X1, vx, vy, vz, th) 

!    !$omp parallel do if (iend-ista.ge.nth) private (j,k)
!    DO i = ista,iend
!    !$omp parallel do if (iend-ista.lt.nth) private (k)
!       DO j = 1,ny
!          DO k = 1,nz

!             INCLUDE 'include/bouss/bouss_rkstep1.f90'

!          END DO
!       END DO
!    END DO

!    ! Runge-Kutta step 2
!    ! Evolves the system in time

!    DO o = ord,1,-1
!       INCLUDE 'include/bouss/bouss_rkstep2.f90'
   
!    END DO


! CALL ThreeTo1D(X2, vx, vy, vz, th) 

! !Save derivative in variable f_X
! CALL Deriv_X(f_X0, X1,X2, X0, dt)

! CALL Norm(aux_norm,f_X0)
! IF (myrank.eq.0) THEN
! print '(A,es10.3)', 'Fin diff: |f_X0| = ',aux_norm
! ENDIF

! pr = pr0

! !Recover fields at initial time
! CALL OneTo3D(vx, vy, vz, th, X0)

! INCLUDE 'include/bouss/bouss_deriv.f90' 
! CALL ThreeTo1D(f_Y, vx, vy, vz, th) 

! CALL Norm(aux_norm,f_Y)
! IF (myrank.eq.0) THEN
! print '(A,es10.3)', 'RHS: |f_X0_| = ',aux_norm
! ENDIF

! Res = f_Y - f_X0

! CALL Norm(aux_norm,Res)
! IF (myrank.eq.0) THEN
! print '(A,es10.3)', '|f_X0-f_X0_| = ',aux_norm
! ENDIF


! pr = pr0

! !Recover fields at initial time
! CALL OneTo3D(vx, vy, vz, th, X0)


! !performs evolution of vx, vy, vz, th in time T
! INCLUDE 'include/bouss/bouss_evol_T.f90' 

! !Transforms to a 1d variable X_evol
! CALL ThreeTo1D(X_evol, vx, vy, vz, th)

! CALL Norm(aux_norm,X_evol)
! IF (myrank.eq.0) THEN
! print '(A,es10.3)', '|X_evol| = ',aux_norm
! ENDIF

! pr = pr0

! !Calculate the derivative in time T
!    !$omp parallel do if (iend-ista.ge.nth) private (j,k)
!    DO i = ista,iend
!    !$omp parallel do if (iend-ista.lt.nth) private (k)
!       DO j = 1,ny
!          DO k = 1,nz

!             INCLUDE 'include/bouss/bouss_rkstep1.f90'

!          END DO
!       END DO
!    END DO

!    ! Runge-Kutta step 2
!    ! Evolves the system in time

!    DO o = ord,1,-1
!       INCLUDE 'include/bouss/bouss_rkstep2.f90'
   
!    END DO

! ! INCLUDE 'include/bouss/bouss_deriv.f90' 

! CALL ThreeTo1D(X1, vx, vy, vz, th) 

!    !$omp parallel do if (iend-ista.ge.nth) private (j,k)
!    DO i = ista,iend
!    !$omp parallel do if (iend-ista.lt.nth) private (k)
!       DO j = 1,ny
!          DO k = 1,nz

!             INCLUDE 'include/bouss/bouss_rkstep1.f90'

!          END DO
!       END DO
!    END DO

!    ! Runge-Kutta step 2
!    ! Evolves the system in time

!    DO o = ord,1,-1
!       INCLUDE 'include/bouss/bouss_rkstep2.f90'
   
!    END DO

! ! INCLUDE 'include/bouss/bouss_deriv.f90' 

! CALL ThreeTo1D(X2, vx, vy, vz, th) 

! !Save derivative in variable f_Y
! CALL Deriv_X(f_Y, X1,X2, X_evol, dt)

! CALL Norm(aux_norm,f_Y)
! IF (myrank.eq.0) THEN
! print '(A,es10.3)', 'Fin diff.: |f_Y| = ',aux_norm
! ENDIF


! !Recover fields at time t+T
! CALL OneTo3D(vx, vy, vz, th, X_evol)

! INCLUDE 'include/bouss/bouss_deriv.f90' 
! CALL ThreeTo1D(f_X0, vx, vy, vz, th) 

! CALL Norm(aux_norm,f_X0)
! IF (myrank.eq.0) THEN
! print '(A,es10.3)', 'RHS: |f_Y_| = ',aux_norm
! ENDIF

! Res = f_Y - f_X0

! CALL Norm(aux_norm,Res)
! IF (myrank.eq.0) THEN
! print '(A,es10.3)', '|f_Y-f_Y_| = ',aux_norm
! ENDIF

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!! End time deriv check
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!Save initial field in 1d variable X0
CALL ThreeTo1D(X0, vx, vy, vz, th) 

!TODO: check if making a copy of original p makes a difference
pr0 = pr

CALL divergence(vx,vy,vz,div)
CALL Norm(aux_norm, X0)
IF (myrank.eq.0) THEN
OPEN(1,file='debug_gmres.txt',position='append')
WRITE(1,FMT='(A,es10.3,A,es10.3)') 'div(X0)=',div,'  |X0|=',aux_norm
CLOSE(1)
ENDIF

! INCLUDE 'include/bouss/bouss_deriv.f90' 
! CALL ThreeTo1D(f_X0, vx, vy, vz, th) 

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!Compute time derivative with finite differences
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!TODO1: check if making dt smaller for this is better, if possible
!TODO2: check if using 3 points finite difference is better
!Evolve dt 
   !TODO: Copy is made just before entering bouss_newt but i leave it for now
   !$omp parallel do if (iend-ista.ge.nth) private (j,k)
   DO i = ista,iend
   !$omp parallel do if (iend-ista.lt.nth) private (k)
      DO j = 1,ny
         DO k = 1,nz

            INCLUDE 'include/bouss/bouss_rkstep1.f90'

         END DO
      END DO
   END DO

   ! Runge-Kutta step 2
   ! Evolves the system in time

   DO o = ord,1,-1
      INCLUDE 'include/bouss/bouss_rkstep2.f90'
   
   END DO

CALL ThreeTo1D(X1, vx, vy, vz, th) 

!Save derivative in variable f_X
CALL Deriv_X(f_X0, X1, X0, dt)

!!!!!!!!!! End of finite difference


CALL OneTo3D(vx,vy,vz,th,f_X0)

CALL divergence(vx,vy,vz,div)
CALL Norm(aux_norm, f_X0)
IF (myrank.eq.0) THEN
OPEN(1,file='debug_gmres.txt',position='append')
WRITE(1,FMT='(A,es10.3,A,es10.3)') 'div(f(X0))=',div,'   |f(X0)|=',aux_norm
CLOSE(1)
ENDIF


CALL OneTo3D(vx,vy,vz,th,X0)
pr = pr0
!performs evolution of vx, vy, vz, th in time T
INCLUDE 'include/bouss/bouss_evol_T.f90' 
CALL ThreeTo1D(X_evol, vx, vy, vz, th) 

CALL divergence(vx,vy,vz,div)
CALL Norm(aux_norm, X_evol)
IF (myrank.eq.0) THEN
OPEN(1,file='debug_gmres.txt',position='append')
WRITE(1,FMT='(A,es10.3,A,es10.3)') 'div(X(t+T))=',div,'   |X(t+T)|=',aux_norm
CLOSE(1)
ENDIF

! INCLUDE 'include/bouss/bouss_deriv.f90' 
! CALL ThreeTo1D(f_Y, vx, vy, vz, th) 

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!Compute time derivative with finite differences
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!TODO1: check if making dt smaller for this is better, if possible
!TODO2: check if using 3 points finite difference is better
!Evolve dt 
   !TODO: Copy is made just before entering bouss_newt but i leave it for now
   !$omp parallel do if (iend-ista.ge.nth) private (j,k)
   DO i = ista,iend
   !$omp parallel do if (iend-ista.lt.nth) private (k)
      DO j = 1,ny
         DO k = 1,nz

            INCLUDE 'include/bouss/bouss_rkstep1.f90'

         END DO
      END DO
   END DO

   ! Runge-Kutta step 2
   ! Evolves the system in time

   DO o = ord,1,-1
      INCLUDE 'include/bouss/bouss_rkstep2.f90'
   
   END DO

CALL ThreeTo1D(X1, vx, vy, vz, th) 

!Save derivative in variable f_X
CALL Deriv_X(f_Y, X1, X_evol, dt)

!!!!!!!!!! End of finite difference

CALL divergence(vx,vy,vz,div)
CALL Norm(aux_norm, f_Y)
IF (myrank.eq.0) THEN
OPEN(1,file='debug_gmres.txt',position='append')
WRITE(1,FMT='(A,es10.3,A,es10.3)') 'div(f(Y))=',div,'   |f(Y)|=',aux_norm
CLOSE(1)
ENDIF

!Traslate in the guessed shifts:
CALL Translation(Y0, X_evol, 1, sx) 
CALL Translation(Y0, Y0, 2, sy) 

CALL Translation(f_Y, f_Y, 1, sx) 
CALL Translation(f_Y, f_Y, 2, sy)

!Calculate RHS (-F(X0, T, sx ,sy ))
!$omp parallel do
DO i = 1,n_dim_1d
   b(i) = X0(i) - Y0(i) 
ENDDO

CALL Norm(b_norm,b)
IF (myrank.eq.0) THEN
print '(A,es10.3)', '|b|=',b_norm
ENDIF

Res = b/b_norm
Q(:,1) = Res

dT_guess= CMPLX(0.0_GP, 0.0_GP, KIND = GP)
d_sx= CMPLX(0.0_GP, 0.0_GP, KIND = GP)
d_sy= CMPLX(0.0_GP, 0.0_GP, KIND = GP)

dX = Res

DO i=1,3
Q_aux(i,1) = CMPLX(0.0_GP,0.0_GP, KIND = GP)
ENDDO

e(1) = 1.0_GP
beta(1) = b_norm

GMRES: DO n = 1,n_max
   
   !!! We then compute the LHS of (1) term by term:

   !Calculate the directional derivative
   CALL Perturb(X_pert,epsilon,X0,dX )
   !TODO: check if perturbing with non solenoidal field is troublesome
   CALL OneTo3D(vx, vy, vz, th, X_pert)

   !pr field used for boundary conditions not what it should
   pr = pr0
   INCLUDE 'include/bouss/bouss_evol_T.f90' 

   !Transforms to a 1d variable X_evol
   CALL ThreeTo1D(X_pert_evol, vx, vy, vz, th) 

   !Calculates the directional derivative term    
   CALL X_fin_diff(X_partial_dif, X_pert_evol, Y0, sx, sy, epsilon)

   !Computes terms of translation generator:
   CALL Shift_term(Y_shift_x, Y0, 1, d_sx) 
   CALL Shift_term(Y_shift_y, Y0, 2, d_sy) 

   !!! Now for the rest of the first column of A:

   !Calculates the projection along the shifted directions and along the direction of flow

   CALL CalculateProjection(proj_f, proj_x, proj_y, dX, X0, f_X0)

   CALL Norm(aux_norm, X_partial_dif)

   IF (myrank.eq.0) THEN
   OPEN(1,file='debug_gmres.txt',position='append')
   WRITE(1,FMT='(A)') ''
   WRITE(1,FMT='(A, I3,A, I3)') 'GMRes iteration: ', n, ' Newton iteration: ', n_newt
   WRITE(1,FMT='(A,es10.3)') '|X_partial_dif|=',aux_norm
   WRITE(1,FMT='(A,es10.3,A,es10.3)') 'REAL(proj_f)=',REAL(proj_f), ' + IM(proj_f)=',AIMAG(proj_f)
   WRITE(1,FMT='(A,es10.3,A,es10.3)') 'REAL(proj_x)=',REAL(proj_x), ' + IM(proj_x)=',AIMAG(proj_x)
   WRITE(1,FMT='(A,es10.3,A,es10.3)') 'REAL(proj_y)=',REAL(proj_y), ' + IM(proj_y)=',AIMAG(proj_y)
   CLOSE(1)
   ENDIF

   ! Can now form r_n = A*r_(n-1)
   CALL Form_Res(Res, Res_aux, dX, X_partial_dif, f_Y, dT_guess ,Y_shift_x, Y_shift_y, &
   proj_f, proj_x, proj_y)


   CALL Arnoldi_step(Res, Res_aux, Q, Q_aux, H, res_norm, n)

   CALL Update_values(dX, d_sx, d_sy, dT_guess, Res, Res_aux)

   IF (myrank.eq.0) THEN
   OPEN(1,file='debug_gmres.txt',position='append')
   WRITE(1,FMT='(A,es10.3,A,es10.3)') 'REAL(d_sx)=',REAL(d_sx), ' + IM(d_sx)=',AIMAG(d_sx)
   WRITE(1,FMT='(A,es10.3,A,es10.3)') 'REAL(d_sy)=',REAL(d_sy), ' + IM(d_sy)=',AIMAG(d_sy)
   WRITE(1,FMT='(A,es10.3,A,es10.3)') 'REAL(dT_guess)=',REAL(dT_guess), ' + IM(dT_guess)=',AIMAG(dT_guess)
   CLOSE(1)
   ENDIF

   CALL Givens_rotation(H, cs, sn, n)

   CALL Update_error(beta, cs, sn, e, b_norm, res_norm, n)

   IF (myrank.eq.0) THEN
   OPEN(1,file='error_gmres.txt',position='append')
   WRITE(1,FMT='(I3,A,es10.3)') n, ': ',e(n+1)
   CLOSE(1)
   ENDIF

   IF (e(n+1)<tol) THEN
      IF (myrank.eq.0) THEN
      print *, 'GMRES Convergence'
      ENDIF
      n_last = n
      EXIT GMRES
   ENDIF

END DO GMRES

IF (myrank.eq.0) THEN
print *, ''
print *, 'Exited GMRES'
print '(A,es10.3)', 'Final error:',e(n_last+1)
ENDIF

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!  Line search !!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! IF (myrank.eq.0) THEN
! print '(A,es10.3)', 'T_guess= ', T_guess
! print '(A,es10.3)', 'sx= ', sx
! print '(A,es10.3)', 'sy= ',sy
! print '(A,es10.3)', '|X0|= ',aux_norm 
! ENDIF

! ! lambda = 10.0_GP**(-6)
! lambda = 1.0_GP

! Hookstep: DO n = 1,n_max_hook
!    IF (myrank.eq.0) THEN
!    print *, ''
!    print '(A,I2)', 'Hookstep iter:',n
!    print '(A,es10.3)', 'lambda=',lambda
!    ENDIF

!    CALL Backpropagation(y_sol, H, beta)
   
!    y_sol = y_sol*lambda

!    !Compute F(X+dX) to check if |F(X+dX)|<=|F(X)|

!    CALL Update_values2(dX, sx, sy, T_guess,d_sx, d_sy, dT_guess, Q, Q_aux, y_sol)

!    !!!!!!!Solenoidal Projection!!!!!!!!!
!    !We apply the projection to dX rather than X+dX so that the numerical error is reduced
!    !Previous attempts at projecting X+dX yielded a field of norm ten times greater than 
!    !X or dX, although we don't know exactly why
!    CALL OneTo3D(vx, vy, vz, th, dX)
!    ! Apply boundary conditions and project onto solenoidal space
!    o = 1
!    CALL v_imposebc_and_project(vplanbc,planfc,vx,vy,vz,pr,rki=o,&
!             v_zsta=(/ vxzsta, vyzsta/), v_zend=(/ vxzend, vyzend/))
!    CALL s_imposebc(splanbc,planfc,th)
!    CALL fc_filter(th)

!    ! Hack to prevent weird accumulation in complex domain for th
!    CALL fftp3d_complex_to_real(planfc,th,R1,MPI_COMM_WORLD)
!    R1 = R1/nx/ny/nz
!    CALL fftp3d_real_to_complex(planfc,R1,th,MPI_COMM_WORLD)

!    CALL ThreeTo1D(dX, vx, vy, vz, th)
!    X_pert = X0+dX

!    CALL OneTo3D(vx, vy, vz, th, X_pert)

!    CALL Norm(aux_norm,X_pert)
!    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!    !!!!!!!!!!!!! F(X+dX)<F(X) !!!!!!!!!!!!!!!!!!
!    !performs evolution of vx, vy, vz, th in time T
!    !TODO: boundary conditions will not match the pressure that has been computed for time T
!    !Could be solved with an auxiliary pr_aux
!    INCLUDE 'include/bouss/bouss_evol_T.f90' 

!    !Transforms to a 1d variable X_evol
!    CALL ThreeTo1D(X_evol, vx, vy, vz, th)

!    !Traslate in the guessed shifts:
!    CALL Translation(Y0, X_evol, 1, sx) 
!    CALL Translation(Y0, Y0, 2, sy) 

!    dX = X_pert - Y0

!    CALL Norm(aux_norm,dX)

!    IF (myrank.eq.0) THEN
!    print '(A,es10.3)', '|b_new|=',aux_norm
!    print '(A,es10.3)', '|b_old|=',b_norm
!    ENDIF

!    IF (aux_norm.le.b_norm) THEN
!       EXIT
!    ELSE
!       lambda = lambda*0.5_GP
!       T_guess = T_guess - REAL(dT_guess)
!       sx = sx - REAL(d_sx)
!       sy = sy - REAL(d_sy)
!    ENDIF
! END DO Hookstep

! CALL OneTo3D(vx, vy, vz, th, X_pert)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!  End of Line search !!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!  Trust Region !!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


! Hookstep: moving the solution to the "trust region"
! Delta will delimit our trust region (i.e. |y_sol|<Delta).
! In the inner loop (j) we adjust mu_hook s.t. |y_sol|<Delta.
! Initially we increment mu_hook in steps of 1, but it can be improved.
! In the outer loop (i) once the inner loop is done we check if
! the linearized problem is valid, i.e. we check |F(X+dX)|<=|F(X)|.
! For this purpose X+dX must be evolved and translated. If
! the condition does not hold Delta must be reduced (reduce
! the trust region).

CALL Backpropagation(y_sol, H, beta)
y_norm = SQRT(SUM(ABS(y_sol)**2))
Delta = y_norm

IF (myrank.eq.0) THEN
print '(A,es10.3)', 'T_guess= ', T_guess
print '(A,es10.3)', 'sx= ', sx
print '(A,es10.3)', 'sy= ',sy
print '(A,es10.3)', '|X0|= ',X0_norm 
ENDIF

IF (myrank.eq.0) THEN
print *, '' 
print '(A,es10.3)', 'Initial Delta=|X0|*10^(-5)=', Delta
print '(A,es10.3)', 'REAL(H(1,1))=',REAL(H(1,1))
ENDIF
Hookstep: DO n = 1,n_max_hook !TODO: check if a new variable is preferable (n_max_hook) 
   IF (myrank.eq.0) THEN
   print *, ''
   print '(A,I2)', 'Hookstep iter:',n
   print '(A,es10.3)', 'Delta=',Delta
   ENDIF

   CALL Hookstep_iter(y_sol, Delta, H, beta)

   !Compute F(X+dX) to check if |F(X+dX)|<=|F(X)|

   CALL Update_values2(dX, sx, sy, T_guess,d_sx, d_sy, dT_guess, Q, Q_aux, y_sol)

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   !!!!!!!Solenoidal Projection!!!!!!!!!
   !We apply the projection to dX rather than X+dX so that the numerical error is reduced
   !Previous attempts at projecting X+dX yielded a field of norm ten times greater than 
   !X or dX, although we don't know exactly why
   CALL OneTo3D(vx, vy, vz, th, dX)

   ! Apply boundary conditions and project onto solenoidal space
   o = -1

   CALL v_imposebc_and_project(vplanbc,planfc,vx,vy,vz,pr,rki=o,&
            v_zsta=(/ vxzsta, vyzsta/), v_zend=(/ vxzend, vyzend/))
   ! CALL sol_project(vx,vy,vz,pr,1,0,0)
   CALL s_imposebc(splanbc,planfc,th)
   CALL fc_filter(th)

   ! Hack to prevent weird accumulation in complex domain for th
   CALL fftp3d_complex_to_real(planfc,th,R1,MPI_COMM_WORLD)
   R1 = R1/nx/ny/nz
   CALL fftp3d_real_to_complex(planfc,R1,th,MPI_COMM_WORLD)

   CALL ThreeTo1D(dX, vx, vy, vz, th)
   !!!!!! End Solenoidal Projection
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   X_pert = X0+dX

   CALL OneTo3D(vx, vy, vz, th, dX)

   CALL divergence(vx,vy,vz,div)
   CALL Norm(aux_norm, dX)
   IF (myrank.eq.0) THEN
   OPEN(1,file='debug_gmres.txt',position='append')
   WRITE(1,FMT='(A, I3,A,es10.3,A,es10.3)') 'Hookstep iter=', n, 'div(dX)=',div, '   |dX|=', aux_norm
   CLOSE(1)
   ENDIF

   CALL OneTo3D(vx, vy, vz, th, X_pert)
   CALL divergence(vx,vy,vz,div)
   CALL Norm(aux_norm, X_pert)
   IF (myrank.eq.0) THEN
   OPEN(1,file='debug_gmres.txt',position='append')
   WRITE(1,FMT='(A, I3,A,es10.3,A,es10.3)') 'Hookstep iter=', n, 'div(X_pert)=',div, '    |X_pert|=', aux_norm
   CLOSE(1)
   ENDIF


   !!!!!!!!!!!!! F(X+dX)<F(X) !!!!!!!!!!!!!!!!!!
   !performs evolution of vx, vy, vz, th in time T
   !TODO: boundary conditions will not match the pressure that has been computed for time T
   !Could be solved with an auxiliary pr_aux
   INCLUDE 'include/bouss/bouss_evol_T.f90' 

   !Transforms to a 1d variable X_evol
   CALL ThreeTo1D(X_evol, vx, vy, vz, th)

   CALL divergence(vx,vy,vz,div)
   CALL Norm(aux_norm, X_evol)
   IF (myrank.eq.0) THEN
   OPEN(1,file='debug_gmres.txt',position='append')
   WRITE(1,FMT='(A, I3,A,es10.3,A,es10.3)') 'Hookstep iter=', n, 'div(X_evol)=',div, '    |X_evol|=', aux_norm
   CLOSE(1)
   ENDIF

   !Traslate in the guessed shifts:
   CALL Translation(Y0, X_evol, 1, sx) 
   CALL Translation(Y0, Y0, 2, sy) 

   dX = X_pert - Y0

   CALL Norm(b_new_norm,dX)

   IF (myrank.eq.0) THEN
   print '(A,es10.3)', '|b_new|=',b_new_norm
   print '(A,es10.3)', '|b_old|=',b_norm
   ENDIF

   IF (b_new_norm.le.b_norm) THEN
      EXIT
   ELSE
      Delta = Delta*0.5_GP
      T_guess = T_guess - REAL(dT_guess)
      sx = sx - REAL(d_sx)
      sy = sy - REAL(d_sy)
      ! IF (b_new_norm.le.b_norm) THEN
      ! b_prev_norm = b_new_norm
      ! ENDIF
   ENDIF
END DO Hookstep
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!! Save the fields at time t+T (snapshot pair)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CALL OneTo3D(vx, vy, vz, th, Y0)
!$omp parallel do if (iend-ista.ge.nth) private (j,k)

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
CALL io_write(1,odir,'vx_T',ext,planio,R1)
CALL io_write(1,odir,'vy_T',ext,planio,R2)
CALL io_write(1,odir,'vz_T',ext,planio,R3)
!$omp parallel do if (iend-ista.ge.nth) private (j,k)
DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
   DO j = 1,ny
      DO k = 1,nz
         C1(k,j,i) = th(k,j,i)*rmp
      END DO
   END DO
END DO
CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
CALL io_write(1,odir,'th_T',ext,planio,R1)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!! End file save
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CALL OneTo3D(vx, vy, vz, th, X_pert)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!  End of Trust Region !!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!Without Hookstep!!!!!!!!!!!!!!
! CALL Backpropagation(y_sol, H, beta)
! y_norm = SQRT(SUM(ABS(y_sol)**2))
! CALL Norm(aux_norm,X0)

! IF (myrank.eq.0) THEN
! print *, 'T_guess=', T_guess, 'sx=', sx, 'sy',sy
! print *, '|X0|=',aux_norm 
! ENDIF

! CALL Update_values2(vx, vy, vz, th, X_pert,dX,d_sx, d_sy, dT_guess, X0, Q, Q_aux, y_sol)

! ! Apply boundary conditions and project onto solenoidal space
! o = 1
! CALL v_imposebc_and_project(vplanbc,planfc,vx,vy,vz,pr,rki=o,&
!          v_zsta=(/ vxzsta, vyzsta/), v_zend=(/ vxzend, vyzend/))
! CALL s_imposebc(splanbc,planfc,th)
! CALL fc_filter(th)

! ! Hack to prevent weird accumulation in complex domain for th
! CALL fftp3d_complex_to_real(planfc,th,R1,MPI_COMM_WORLD)
! R1 = R1/nx/ny/nz
! CALL fftp3d_real_to_complex(planfc,R1,th,MPI_COMM_WORLD)

! T_guess = T_guess + dT_guess
! sx = sx + d_sx
! sy = sy + d_sy

! CALL Norm(aux_norm,dX)


! IF (myrank.eq.0) THEN
! print *, 'T_guess+dT=', T_guess, 'sx+ds=', sx, 'sy+ds',sy
! print *, '|dX|=',aux_norm, '|y_sol|=',y_norm
! ENDIF

! CALL Norm(aux_norm,X_pert)
! IF (myrank.eq.0) THEN
! print *, '|X0+dX|=',aux_norm 
! ENDIF
