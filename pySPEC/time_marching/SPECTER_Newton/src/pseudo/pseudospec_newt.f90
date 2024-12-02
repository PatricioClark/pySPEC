!=================================================================
! PSEUDOSPECTRAL subroutines
!
! Subroutines to do Newton and stabilized biconjugate gradient
! methods to prepare data for the GPE solver. Can be easily
! modified to prepare initial conditions close to a fixed
! point for any solver. You should use the FFTPLANS and
! MPIVARS modules (see the file 'fftp_mod.f90') in each 
! program that calls any of the subroutines in this file. 
!
! NOTATION: index 'i' is 'x' 
!           index 'j' is 'y'
!           index 'k' is 'z'
!
! 2015 P.D. Mininni and M.E. Brachet
!
! 17 May 2018: Support for elongated box (N.Muller & P.D.Mininni) 
!=================================================================



!*****************************************************************
      SUBROUTINE FlattenFields(xvec1d,vx,vy,vz,th)
!-----------------------------------------------------------------
!
! Copies the data into a 1D real array to do the Newton method.
!
! Parameters
!     a     : x component
!     b     : y component
!     c     : z component
!     d     : thermal component
!     xvec1d  : 1D vector
!
      USE fprecision
      USE newtmod
      USE grid
      USE mpivars
      USE commtypes
      USE fft
      USE iovar

!$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_dim_1d)          :: xvec1D
      COMPLEX(KIND=GP), INTENT(IN), DIMENSION(nz,ny,ista:iend) :: vx, vy, vz, th
      COMPLEX(KIND=GP), ALLOCATABLE, DIMENSION(:,:,:) :: C1, C2, C3, C4
      REAL(KIND=GP), ALLOCATABLE, DIMENSION(:,:,:) :: R1, R2, R3, Rth
      INTEGER             :: i,j,k
      REAL(KIND=GP)       :: rmp
      INTEGER             :: offset1,offset2

      ALLOCATE( C1(nz,ny,ista:iend), C2(nz,ny,ista:iend), C3(nz,ny,ista:iend), C4(nz,ny,ista:iend) )
      ALLOCATE( R1(nx,ny,ksta:kend), R2(nx,ny,ksta:kend), R3(nx,ny,ksta:kend), Rth(nx,ny,ksta:kend) )


      rmp = 1.0_GP/ &
            (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))
!$omp parallel do if (iend-ista.ge.nth) private (j,k)
      DO i = ista,iend
!$omp parallel do if (iend-ista.lt.nth) private (k)
            DO j = 1,ny
            DO k = 1,nz
                  C1(k,j,i) = vx(k,j,i)*rmp
                  C2(k,j,i) = vy(k,j,i)*rmp
                  C3(k,j,i) = vz(k,j,i)*rmp
                  C4(k,j,i) = th(k,j,i)*rmp
            END DO
            END DO
      END DO
      CALL fftp3d_complex_to_real(planfc,C1,R1,MPI_COMM_WORLD)
      CALL fftp3d_complex_to_real(planfc,C2,R2,MPI_COMM_WORLD)
      CALL fftp3d_complex_to_real(planfc,C3,R3,MPI_COMM_WORLD) 
      CALL fftp3d_complex_to_real(planfc,C4,Rth,MPI_COMM_WORLD)

!$omp parallel do if ((iend-ista).ge.nth) private(j,k,offset1,offset2)
      DO k = ksta,kend_ph
         offset1 = 4*(k-ksta)*nx*ny
!$omp parallel do if ((iend-ista).lt.nth) private(k,offset2)
         DO j = 1,ny
            offset2 = offset1 + 4*(j-1)*nx
            DO i = 1,nx
               xvec1D(1+4*(i-1)+offset2) = R1(i,j,k)
               xvec1D(2+4*(i-1)+offset2) = R2(i,j,k)
               xvec1D(3+4*(i-1)+offset2) = R3(i,j,k)
               xvec1D(4+4*(i-1)+offset2) = Rth(i,j,k)
            END DO
         END DO
      END DO

      DEALLOCATE(R1, R2, R3, Rth)
      DEALLOCATE(C1, C2, C3, C4)

      RETURN
      END SUBROUTINE FlattenFields


!*****************************************************************
      SUBROUTINE UnflattenFields(vx,vy,vz,th,xvec1d)
!-----------------------------------------------------------------
!
! Converts 1d real array into complex fields.
!
! Parameters
!     a     : x component
!     b     : y component
!     c     : z component
!     d     : thermal component
!     xvec1d  : 1D vector
!
      USE fprecision
      USE mpivars
      USE newtmod
      USE grid
      USE commtypes
      USE fft

!$    USE threads
      IMPLICIT NONE

      COMPLEX(KIND=GP), INTENT(OUT), DIMENSION(nz,ny,ista:iend) :: vx,vy,vz,th
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d)          :: xvec1D
      REAL(KIND=GP), ALLOCATABLE, DIMENSION(:,:,:) :: R1, R2, R3, Rth
      INTEGER             :: i,j,k
      REAL(KIND=GP)       :: rmp
      INTEGER             :: offset1,offset2

      ALLOCATE( R1(nx,ny,ksta:kend), R2(nx,ny,ksta:kend), R3(nx,ny,ksta:kend), Rth(nx,ny,ksta:kend) )

!$omp parallel do if ((iend-ista).ge.nth) private(j,k,offset1,offset2)
      DO k = ksta,kend_ph
         offset1 = 4*(k-ksta)*nx*ny
!$omp parallel do if ((iend-ista).lt.nth) private(k,offset2)
         DO j = 1,ny
            offset2 = offset1 + 4*(j-1)*nx
            DO i = 1,nx
               R1(i,j,k) = xvec1D(1+4*(i-1)+offset2)
               R2(i,j,k) = xvec1D(2+4*(i-1)+offset2)
               R3(i,j,k) = xvec1D(3+4*(i-1)+offset2)
               Rth(i,j,k) = xvec1D(4+4*(i-1)+offset2)
            END DO
         END DO
      END DO

      CALL fftp3d_real_to_complex(planfc,R1,vx,MPI_COMM_WORLD)
      CALL fftp3d_real_to_complex(planfc,R2,vy,MPI_COMM_WORLD)
      CALL fftp3d_real_to_complex(planfc,R3,vz,MPI_COMM_WORLD)
      CALL fftp3d_real_to_complex(planfc,Rth,th,MPI_COMM_WORLD)

      DEALLOCATE(R1, R2, R3, Rth)

      RETURN
      END SUBROUTINE UnflattenFields

!*****************************************************************
      SUBROUTINE Translation(xvec1D_out, xvec1D, direc, s)
!-----------------------------------------------------------------
!
! Translate fields contained in xvec1D by the amount "s" in the 
! direction "direc", where 1 = x and 2 = y.
! 
! Parameters
!     xvec1D_out  : 1D output vector
!     xvec1D  : 1D input vector
!     direc  : direction in which to translate

      USE fprecision
      USE newtmod
      USE mpivars
      USE grid
      USE kes
      USE var
   !$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_dim_1d) :: xvec1D_out 
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: xvec1D 
      COMPLEX(KIND=GP), ALLOCATABLE, DIMENSION(:,:,:) :: a,b,c,d
      INTEGER, INTENT(IN) :: direc !if direc = 1 traslates in x, if =2 traslates in y
      INTEGER :: i, j, k
      REAL(KIND=GP), INTENT(IN) :: s

      ALLOCATE( a(nz,ny,ista:iend), b(nz,ny,ista:iend), c(nz,ny,ista:iend), d(nz,ny,ista:iend) )

      CALL UnflattenFields(a, b, c, d, xvec1D)

      IF (direc.eq.1) THEN
            !$omp parallel do if ((iend-ista).ge.nth) private(j,k,offset1,offset2)
            DO i = ista,iend
                  !$omp parallel do if ((iend-ista).lt.nth) private(k,offset2)
                  DO j = 1,ny
                        DO k = 1,nz
                              a(k,j,i) = a(k,j,i) * EXP( im * kx(i)*s) 
                              b(k,j,i) = b(k,j,i) * EXP( im * kx(i)*s)
                              c(k,j,i) = c(k,j,i) * EXP( im * kx(i)*s)
                              d(k,j,i) = d(k,j,i) * EXP( im * kx(i)*s)
                        END DO
                  END DO
            END DO
      ELSE 
            !$omp parallel do if ((iend-ista).ge.nth) private(j,k,offset1,offset2)
            DO i = ista,iend
                  !$omp parallel do if ((iend-ista).lt.nth) private(k,offset2)
                  DO j = 1,ny
                        DO k = 1,nz
                              a(k,j,i) = a(k,j,i) * EXP( im * ky(j)*s) 
                              b(k,j,i) = b(k,j,i) * EXP( im * ky(j)*s)
                              c(k,j,i) = c(k,j,i) * EXP( im * ky(j)*s)
                              d(k,j,i) = d(k,j,i) * EXP( im * ky(j)*s)
                        END DO
                  END DO
            END DO
      ENDIF

      CALL FlattenFields(xvec1D_out, a, b, c, d)

      DEALLOCATE(a, b, c, d)

      RETURN 
      END SUBROUTINE Translation


!*****************************************************************
      SUBROUTINE Scal(s,u1,u2)
!-----------------------------------------------------------------
!
! Routine to compute the reduced scalar product of two 1D
! vectors in double precision (even if GP=SINGLE).
!
! Parameters
!     u1      : First 1D vector
!     u2      : Second 1D vector
!     s       : at the output contais the reduced scalar product
!     n_dim_1d: size of the 1D vectors
!
      USE fprecision
      USE commtypes
      USE newtmod
      USE mpivars
      USE grid
!$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(OUT) :: s
      REAL(KIND=GP)              :: stemp,tmp
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: u1,u2
      INTEGER :: i

      stemp = 0.0D0
      tmp = 1.0_GP/  &
            (real(nx,kind=GP)*real(ny,kind=GP)*real(nz,kind=GP))**2
!$omp parallel do reduction(+:stemp)
      DO i = 1,n_dim_1d
         stemp = stemp+u1(i)*u2(i)!*tmp
      ENDDO
      CALL MPI_ALLREDUCE(stemp,s,1,MPI_DOUBLE_PRECISION,MPI_SUM, &
                         MPI_COMM_WORLD,ierr)

      RETURN
      END SUBROUTINE Scal

!*****************************************************************
      SUBROUTINE Norm(norm_a,a)
!-----------------------------------------------------------------
!
! Routine to compute the Frobenius norm of 1D complex
! vectors in double precision (even if GP=SINGLE).
!
! Parameters
!     norm_a      : Norm
!     a      : 1D complex vector
      USE fprecision
      USE newtmod
!$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(OUT) :: norm_a
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: a
      REAL(KIND=GP) :: aux

      CALL Scal(aux, a, a)
      norm_a =  SQRT(REAL(aux))


      RETURN
      END SUBROUTINE Norm

!*****************************************************************
     SUBROUTINE Perturb(X_pert,epsilon,X0,dX)
!-----------------------------------------------------------------
!
! Computes the 1D vector X_pert by adding a perturbance in the direction
! dX with magnitude such that ||epsilon dX|| = 10^(-7) X0, as stated in 
! Chandler and Kerswell 2012. 
!
! Parameters
!     X_pert      : 1D perturbed vector
!     epsilon      : Perturbance magnitude correction
!     X0      : 1D initial complex vector
!     dX      : Perturbance vector
!

      USE fprecision
      USE newtmod
   !$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_dim_1d) :: X_pert
      REAL(KIND=GP), INTENT(OUT) :: epsilon
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: X0
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: dX
      REAL(KIND=GP) :: norm_X0, norm_dX 
      INTEGER :: i

      CALL Norm(norm_X0,X0)
      CALL Norm(norm_dX,dX)

      epsilon = 10.0_GP**(-7) * norm_X0 / norm_dX

      !$omp parallel do
            DO i = 1,n_dim_1d
            X_pert(i) = X0(i) + epsilon * dX(i)
            ENDDO
      
      RETURN 
      END SUBROUTINE Perturb


!*****************************************************************
     SUBROUTINE X_fin_diff(X_partial_dif, X_evol, Y0, sx, sy, epsilon)
!-----------------------------------------------------------------
!
! Computes the partial derivative of the translated evolved vector with
! respect to the initial vector, times the correction of the initial vector
! dX, using finite differences. 
!
! Parameters
!     X_partial_dif      : 1D output complex vector
!     X_evol      : 1D perturbed vector evolved in T_guess time
!     Y_1d      : 1D unperturbed vector evolved in T_guess time
!     sx, sy      : guessed shifts in x and y
!     epsilon      : Perturbance magnitude correction calculated with Perturb subroutine

      USE fprecision
      USE newtmod
   !$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_dim_1d) :: X_partial_dif
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: X_evol
      REAL(KIND=GP), ALLOCATABLE, DIMENSION(:) :: X_evol_shift
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: Y0
      REAL(KIND=GP), INTENT(IN)    :: sx, sy
      REAL(KIND=GP), INTENT(IN)    :: epsilon
      INTEGER :: i

      ALLOCATE(X_evol_shift(1:n_dim_1d))

      CALL Translation(X_evol_shift, X_evol, 1, sx) !traslado en sx, sy
      CALL Translation(X_evol_shift, X_evol_shift, 2, sy) 

      !$omp parallel do
            DO i = 1,n_dim_1d
            X_partial_dif(i) = (X_evol_shift(i)-Y0(i))/epsilon
            ENDDO

      DEALLOCATE(X_evol_shift)

      RETURN 
      END SUBROUTINE X_fin_diff


!*****************************************************************
     SUBROUTINE Shift_term(Y_shift, Y0, direc, d_s)
!-----------------------------------------------------------------
!
! Computes the shift term by applying the translation infinitesimal
! generators to the evolved shifted Y0, and multiplying by the
! infinitesimal shift d_s in the direction x (1) or y (2)
!
! Parameters
!     Y_shift      : 1D output vector with translational generators applied
!     Y0      : 1D evolved shifted input vector
!     dir   : direction of shift (1=x, 2=y)
!     d_s      : infinitesimal shift in dir. updated by algorithm

      USE fprecision
      USE newtmod
      USE mpivars
      USE grid
      USE kes
      USE var
   !$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_dim_1d) :: Y_shift
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: Y0
      COMPLEX(KIND=GP), ALLOCATABLE, DIMENSION(:,:,:) :: a,b,c,d
      INTEGER, INTENT(IN) :: direc
      REAL(KIND=GP), INTENT(IN) :: d_s
      INTEGER :: i, j, k
      INTEGER :: offset1,offset2
           
      ALLOCATE( a(nz,ny,ista:iend), b(nz,ny,ista:iend), c(nz,ny,ista:iend), d(nz,ny,ista:iend) )

      CALL UnflattenFields(a, b, c, d, Y0)

      IF (direc.eq.1) THEN
            !$omp parallel do if ((iend-ista).ge.nth) private(j,k,offset1,offset2)
            DO i = ista,iend
                  !$omp parallel do if ((iend-ista).lt.nth) private(k,offset2)
                  DO j = 1,ny
                        DO k = 1,nz
                              a(k,j,i) = a(k,j,i) * im * kx(i) * d_s/ Lx
                              b(k,j,i) = b(k,j,i) * im * kx(i) * d_s/ Lx
                              c(k,j,i) = c(k,j,i) * im * kx(i) * d_s/ Lx
                              d(k,j,i) = d(k,j,i) * im * kx(i) * d_s/ Lx
                        END DO
                  END DO
            END DO
      ELSE 
            !$omp parallel do if ((iend-ista).ge.nth) private(j,k,offset1,offset2)
            DO i = ista,iend
                  !$omp parallel do if ((iend-ista).lt.nth) private(k,offset2)
                  DO j = 1,ny
                        DO k = 1,nz
                              a(k,j,i) = a(k,j,i) * im * ky(j) * d_s / Ly 
                              b(k,j,i) = b(k,j,i) * im * ky(j) * d_s / Ly
                              c(k,j,i) = c(k,j,i) * im * ky(j) * d_s / Ly
                              d(k,j,i) = d(k,j,i) * im * ky(j) * d_s / Ly
                        END DO
                  END DO
            END DO
      ENDIF

      CALL FlattenFields(Y_shift, a, b, c, d)

      DEALLOCATE(a, b, c, d)

      RETURN 
      END SUBROUTINE Shift_term


! !*****************************************************************
!      SUBROUTINE f_Y_RB(f_Y, dT_guess)
! !-----------------------------------------------------------------
! !
! ! Computes f(Y0)*dT_guess in Rayleigh Benard flow.
! ! Where f(Y0) = Y0_dot is the time derivative of the state Y0
! ! given by the flow governing equations.
! !
! ! Parameters
! !     Y0  : 1D complex vector
! !     f_Y0  : 1D complex vector

!       USE fprecision
!       USE commtypes
!       USE newtmod
!       USE mpivars
!       USE grid
!    !$    USE threads
!       IMPLICIT NONE

!       COMPLEX(KIND=GP), INTENT(INOUT), DIMENSION(n_dim_1d) :: f_Y
!       REAL(KIND=GP), INTENT(IN) :: dT_guess 
!       INTEGER :: i, j, k
!       INTEGER :: offset1,offset2

!       DO i = ista,iend
!                   f_Y0(1+4*(k-1)+offset2) = (nu*vx(k,j,i)-C4(k,j,i)+fx(k,j,i))*dT_guess
!                   f_Y0(2+4*(k-1)+offset2) = (nu*vy(k,j,i)-C5(k,j,i)+fy(k,j,i))*dT_guess
!                   f_Y0(3+4*(k-1)+offset2) = (nu*vz(k,j,i)-C6(k,j,i)+fz(k,j,i))*dT_guess
!                   f_Y0(4+4*(k-1)+offset2) = (kappa*th(k,j,i)-C8(k,j,i)+fs(k,j,i))*dT_guess
!                   END DO
!             END DO
!       END DO
!       RETURN 
!       END SUBROUTINE f_Y_RB




!*****************************************************************
     SUBROUTINE CalculateProjection(proj_f, proj_x, proj_y, dX, X0, f_X0)
!-----------------------------------------------------------------
!
! Calculates the projection of dX in the directions with traslational
! symmetries such as x and y. It also calculates the components of dX 
! along the direction of f(X0) such that the correction doesnt follow
! the original orbit.
!
! Parameters
!     proj_f, proj_x, proj_y  : scalar product between dX and (respectively) time derivative of X0, and x-y translation generator applied to X0
!     dX  : 1D perturbance vector
!     X0  : initial vector

      USE fprecision
      USE newtmod
   !$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(OUT) :: proj_f, proj_x, proj_y
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: dX, X0, f_X0
      REAL(KIND=GP), ALLOCATABLE ,DIMENSION(:) :: Xtras_x, Xtras_y
      REAL(KIND=GP) :: aux1,aux2
      INTEGER :: i, j, k
      INTEGER :: offset1,offset2

      ALLOCATE(Xtras_x(1:n_dim_1d), Xtras_y(1:n_dim_1d))

      !Projects time derivative with dX (proposed variation to initial field)
      CALL Scal(proj_f, f_X0, dX)
            
      !Computes the initial vector field with the translation generator applied in x and y:

      CALL Shift_term(Xtras_x, X0, 1, 1.0_GP)
      CALL Shift_term(Xtras_y, X0, 2, 1.0_GP)

      !Projects with dX (proposed variation of initial field)
      CALL Scal(proj_x, Xtras_x, dX)
      CALL Scal(proj_y, Xtras_y, dX)

      RETURN 
      END SUBROUTINE CalculateProjection


!*****************************************************************
     SUBROUTINE Form_Res(Res, Res_aux, dX, X_partial_diff, f_Y, dT_guess, Y_shift_x, Y_shift_y, &
      proj_f, proj_x, proj_y)
!-----------------------------------------------------------------
!
! Forms the Residual vector as stated in the GMRES algorithm. In each 
! 
! Parameters
!
!     Res  : 1D residual vector of dim: n_dim_1d
!     Res_aux  : rest of residual vector, dim: 3
!     dX  : 1D perturbance vector
!     X_partial_diff  : Partial derivative term
!     f_Y  : Time derivative term
!     dT_guess  : Shift in T guess
!     Y_shift_x  : x shift term
!     Y_shift_y  : y shift term
!     proj_f, proj_x, proj_y  : scalar product between dX and (respectively) time derivative of X0, and x-y translation generator applied to X0
!     n  : iteration number of GMRES

      USE fprecision
      USE newtmod
   !$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_dim_1d) :: Res
      REAL(KIND=GP), INTENT(OUT), DIMENSION(3) :: Res_aux
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: dX, X_partial_diff, f_Y, Y_shift_x, Y_shift_y
      REAL(KIND=GP), INTENT(IN) :: proj_f, proj_x, proj_y
      REAL(KIND=GP), INTENT(IN) :: dT_guess
      INTEGER :: i

      !$omp parallel do
      DO i = 1,n_dim_1d
      Res(i) = X_partial_diff(i) -dX(i) + Y_shift_x(i) + Y_shift_y(i) + f_Y(i)*dT_guess
      ENDDO
      Res_aux(1) = REAL(proj_x)
      Res_aux(2) = REAL(proj_y)
      Res_aux(3) = REAL(proj_f)

      RETURN 
      END SUBROUTINE Form_Res

!*****************************************************************
     SUBROUTINE Arnoldi_step(Res, Res_aux, Q, Q_aux, H, res_norm, n)
!     -----------------------------------------------------------------
!
! Performs n_th iteration of Arnoldi algorithm, i.e. calculates the new column vector for Q
! using Modified Graham-Schmidt.
! 
! Parameters
!
!     Res  : 1D residual vector of dim: n_dim_1d
!     Res_aux  : rest of residual vector, dim: 3
!     Q  : Arnoldi orthonomal matrix of dim: (n_dim_1d, n)
!     Q_aux  : rest of orthonomal matrix, dim: (3, n)
!     H  : upper hessenberg arnoldi matrix, dim: (n+1,n)
!     n  : iteration number of GMRES

      USE fprecision
      USE newtmod
      USE mpivars
   !$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(INOUT), DIMENSION(n_dim_1d) :: Res
      REAL(KIND=GP), INTENT(INOUT), DIMENSION(3) :: Res_aux
      REAL(KIND=GP), INTENT(INOUT), DIMENSION(n_dim_1d,n_max) :: Q
      REAL(KIND=GP), INTENT(INOUT), DIMENSION(3,n_max) :: Q_aux
      REAL(KIND=GP), INTENT(INOUT), DIMENSION(n_max+1,n_max) :: H
      REAL(KIND=GP), INTENT(OUT) :: res_norm
      REAL(KIND=GP) :: aux
      INTEGER, INTENT(IN) :: n
      INTEGER :: i,j

      DO i = 1, n
            CALL Scal(H(i,n),Q(:,i), Res) !Compute the inner product with the n_dim_1d part
            !Complete the dot product with the aux part
            DO j = 1,3
                  H(i,n) = H(i,n)+Q_aux(j,i)*Res_aux(j)
            ENDDO
            Res = Res - H(i,n) * Q(:,i)
            Res_aux = Res_aux - H(i,n) * Q_aux(:,i)
      END DO

      CALL Norm(res_norm, Res)
      res_norm = SQRT(res_norm**2 + SUM(ABS(Res_aux)**2))
      H(n+1, n) = res_norm
      Res = Res/res_norm
      Res_aux = Res_aux/res_norm
      Q(:,n+1) = Res
      Q_aux(:,n+1) = Res_aux

      RETURN
      END SUBROUTINE Arnoldi_step

!*****************************************************************
     SUBROUTINE Update_values(dX, d_sx, d_sy, dT_guess, Res, Res_aux)
!-----------------------------------------------------------------
!
! Updates the values according to the last arnoldi iteration.
! 
! Parameters
!
!     dX  : 1D perturbance vector
!     d_sx : perturbance in x shift
!     d_sy : perturbance in y shift
!     dT : perturbance in period time
!     Res  : 1D residual vector of dim: n_dim_1d
!     Res_aux  : rest of residual vector, dim: 3
!
      USE fprecision
      USE newtmod
   !$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_dim_1d) :: dX
      REAL(KIND=GP), INTENT(OUT) :: d_sx, d_sy, dT_guess
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d) :: Res
      REAL(KIND=GP), INTENT(IN), DIMENSION(3) :: Res_aux
      INTEGER :: i

      !$omp parallel do
      DO i = 1,n_dim_1d
      dX(i) = Res(i)
      ENDDO

      d_sx = Res_aux(1)
      d_sy = Res_aux(2)
      dT_guess = Res_aux(3)

      RETURN
      END SUBROUTINE Update_values


!*****************************************************************
     SUBROUTINE Givens_rotation(H, cs, sn, n)
!-----------------------------------------------------------------
!
! Performs Givens rotation
!
      USE fprecision
      USE newtmod
      USE mpivars
   !$    USE threads
      IMPLICIT NONE

      REAL(KIND=GP), INTENT(INOUT), DIMENSION(n_max+1,n_max) :: H
      REAL(KIND=GP), INTENT(INOUT), DIMENSION(n_max) :: cs, sn
      REAL(KIND=GP) :: temp
      REAL(KIND=GP) :: hip
      INTEGER, INTENT(IN) :: n
      INTEGER :: i

      !Premultiply the last H column (n) by the previous n-1 Givens matrices
      DO i = 1, n-1
            temp = cs(i)*H(i,n) + sn(i)*H(i+1,n)
            H(i+1,n) = -sn(i)*H(i,n) + cs(i)*H(i+1,n)
            H(i,n) = temp
      END DO
      
      ! Find the values of the new cs and sn of the k_th Given matrix
      hip = SQRT(ABS(H(n,n))**2+ABS(H(n+1,n))**2)
      cs(n) = H(n,n)/hip
      sn(n) = H(n+1,n)/hip
      
      !Update the last H entries and eliminate H(n,n-1) to obtain a triangular matrix R
      H(n,n) = cs(n)*H(n,n) + sn(n)*H(n+1,n)
      H(n+1,n) = 0

      RETURN
      END SUBROUTINE Givens_rotation

!*****************************************************************
     SUBROUTINE Update_error(beta, cs, sn, e, b_norm, res_norm, n)
!-----------------------------------------------------------------
!
! Performs Givens rotation
!
      USE fprecision
      USE newtmod
   !$    USE threads
      IMPLICIT NONE

      !TODO: check if complex and real are in right place
      REAL(KIND=GP), INTENT(INOUT), DIMENSION(n_max) :: beta
      REAL(KIND=GP), INTENT(INOUT), DIMENSION(n_max) :: e
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_max) :: cs, sn
      REAL(KIND=GP), INTENT(IN) :: b_norm, res_norm
      REAL(KIND=GP) :: error
      INTEGER, INTENT(IN) :: n
      INTEGER :: i

      beta(n+1) =  -sn(n) * beta(n)
      beta(n) = cs(n) * beta(n)
        
      error = ABS(beta(n+1))/b_norm
      e(n+1) = error

      RETURN
      END SUBROUTINE Update_error

!*****************************************************************
     SUBROUTINE Backpropagation(y_sol, R, b)
!-----------------------------------------------------------------
!
! Performs backpropagation algorithm to solve an upper triangular system of equations R@y=b
! where R is of size (n,n) and b and y of size (n)
!
      USE fprecision
      USE newtmod
   !$    USE threads
      IMPLICIT NONE

      !TODO: check if complex and real are in right place
      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_max) :: y_sol
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_max+1,n_max) :: R
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_max) :: b
      REAL(KIND=GP) :: aux
      INTEGER :: i, j

      DO i = 1, n_max
      y_sol(i) = 0.0_GP
      ENDDO

      DO i = n_last, 1, -1
            aux = 0.0_GP
            DO j = i+1,n_last
                  aux = aux + y_sol(j) * R(i,j)
            END DO
            y_sol(i) = (b(i) - aux)/R(i,i)
      END DO

      RETURN
      END SUBROUTINE Backpropagation

!*****************************************************************
     SUBROUTINE Hookstep_transform(H_hook, H, mu_hook)
!-----------------------------------------------------------------
!
!
      USE fprecision
      USE newtmod
   !$    USE threads
      IMPLICIT NONE
 
      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_max+1,n_max) :: H_hook
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_max+1,n_max) :: H
      REAL(KIND=GP), INTENT(IN) :: mu_hook
      INTEGER :: i

      H_hook = H
      DO i = 1,n_last
            H_hook(i,i) = H(i,i) + mu_hook/H(i,i)
      END DO
      
      RETURN
      END SUBROUTINE Hookstep_transform

!*****************************************************************
     SUBROUTINE Update_values2(dX,sx,sy,T_guess,d_sx, d_sy, dT_guess,&
      Q, Q_aux, y_sol)
!-----------------------------------------------------------------
!
!  x = x0 + Q[:,:n]@y
!
      USE fprecision
      USE newtmod
      USE mpivars
   !$    USE threads
      IMPLICIT NONE
 
      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_dim_1d) :: dX
      REAL(KIND=GP), INTENT(OUT) :: d_sx, d_sy, dT_guess
      REAL(KIND=GP), INTENT(INOUT) :: sx, sy, T_guess
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_dim_1d,n_max) :: Q
      REAL(KIND=GP), INTENT(IN), DIMENSION(3,n_max) :: Q_aux
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_max) :: y_sol
      REAL(KIND=GP) :: aux
      REAL(KIND=GP) :: aux_norm
      INTEGER :: i,j

      DO i = 1,n_dim_1d
            aux = 0.0_GP
            DO j =1,n_last
            aux = aux + Q(i,j)*y_sol(j)
            ENDDO
            dX(i) = aux
      ENDDO

      CALL Norm(aux_norm, dX)

      d_sx = 0.0_GP
      d_sy = 0.0_GP
      dT_guess = 0.0_GP

      DO j =1,n_last
      d_sx = d_sx + Q_aux(1,j)*y_sol(j)
      d_sy = d_sy + Q_aux(2,j)*y_sol(j)
      dT_guess = dT_guess + Q_aux(3,j)*y_sol(j)
      ENDDO

      IF (myrank.eq.0) THEN
      OPEN(1,file='prints/debug_gmres.txt',position='append')
      WRITE(1,FMT='(A)') ''
      WRITE(1,FMT='(A,es10.3)') 'd_sx=',d_sx
      WRITE(1,FMT='(A,es10.3)') 'd_sy=',d_sy
      WRITE(1,FMT='(A,es10.3)') 'dT_guess=',dT_guess
      CLOSE(1)
      ENDIF

      T_guess = T_guess + dT_guess
      sx = sx + d_sx
      sy = sy + d_sy

      IF (myrank.eq.0) THEN
      print '(A,es10.3)', 'T_guess+dT=', T_guess
      print '(A,es10.3)',  'sx+ds=', sx
      print '(A,es10.3)', 'sy+ds=',sy
      print '(A,es10.3)', '|dX|=',aux_norm
      ENDIF

      RETURN
      END SUBROUTINE Update_values2

!*****************************************************************
     SUBROUTINE Hookstep_iter(y_sol, Delta, H, beta)
!-----------------------------------------------------------------
!
!
      USE fprecision
      USE newtmod
      USE mpivars
   !$    USE threads
      IMPLICIT NONE
 
      REAL(KIND=GP), INTENT(OUT), DIMENSION(n_max) :: y_sol
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_max+1,n_max) :: H
      REAL(KIND=GP), INTENT(IN), DIMENSION(n_max) :: beta
      REAL(KIND=GP), INTENT(IN) :: Delta
      REAL(KIND=GP), DIMENSION(n_max+1,n_max) :: H_hook
      REAL(KIND=GP) :: y_norm, mu_hook
      INTEGER :: j

      CALL Backpropagation(y_sol, H, beta)
      y_norm = SQRT(SUM(ABS(y_sol)**2))
      IF (myrank.eq.0) THEN
      print '(A,es10.3)', 'Initial y_norm=',y_norm
      ENDIF
      mu_hook = 0.0_GP
      DO j = 1, 100 
            IF (y_norm.le.Delta) THEN
            EXIT
            ELSE
            mu_hook = 2.0_GP**j !+ max_val/n_max_hook
            CALL Hookstep_transform(H_hook,H,mu_hook)
            CALL Backpropagation(y_sol, H_hook, beta)
            y_norm = SQRT(SUM(ABS(y_sol)**2))
            END IF
      END DO
      IF (myrank.eq.0) THEN
      print '(A,es10.3)', 'Final y_norm=',y_norm
      ENDIF

      RETURN
      END SUBROUTINE Hookstep_iter
      