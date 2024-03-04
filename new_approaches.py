
def main():


    while True: 
        
            if keyword_letter == 113:
                webcamera.release()
                cv2.destroyAllWindows()
                break
                

            # cv2.imshow("Testing the relativeness of pupil center", cv2.flip(frame, 1))
            cv2.imshow("Zero image", zero_image)

            



if __name__ == "__main__":
    main()

