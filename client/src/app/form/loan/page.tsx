"use client"
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import React, { useState } from 'react';

interface CreditApprover {
  CODE_GENDER: string;
  FLAG_OWN_CAR: string;
  FLAG_OWN_REALTY: string;
  CNT_CHILDREN: number;
  AMT_INCOME_TOTAL: number;
  NAME_INCOME_TYPE: string;
  NAME_EDUCATION_TYPE: string;
  NAME_FAMILY_STATUS: string;
  NAME_HOUSING_TYPE: string;
  DAYS_BIRTH: number;
  DAYS_EMPLOYED: number;
  FLAG_MOBIL: number;
  FLAG_WORK_PHONE: number;
  FLAG_PHONE: number;
  FLAG_EMAIL: number;
  OCCUPATION_TYPE: string;
  CNT_FAM_MEMBERS: number;
}

export default function CreditApproverForm() {
  const [formData, setFormData] = useState<CreditApprover>({
    CODE_GENDER: '',
    FLAG_OWN_CAR: '',
    FLAG_OWN_REALTY: '',
    CNT_CHILDREN: 0,
    AMT_INCOME_TOTAL: 0,
    NAME_INCOME_TYPE: '',
    NAME_EDUCATION_TYPE: '',
    NAME_FAMILY_STATUS: '',
    NAME_HOUSING_TYPE: '',
    DAYS_BIRTH: 0,
    DAYS_EMPLOYED: 0,
    FLAG_MOBIL: 0,
    FLAG_WORK_PHONE: 0,
    FLAG_PHONE: 0,
    FLAG_EMAIL: 0,
    OCCUPATION_TYPE: '',
    CNT_FAM_MEMBERS: 0,
  });

  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
    setFormErrors({
      ...formErrors,
      [name]: '', // Clear the error when the user starts typing
    });
  };

  const validateForm = () => {
    const errors: Record<string, string> = {};

    // Add your validation logic here
    // For example, check if required fields are filled, validate email format, etc.

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (validateForm()) {
      // You can handle form submission logic here, for example, sending data to a server.
      console.log('Form submitted:', formData);
    } else {
      console.log('Form has errors. Please fix them before submitting.');
    }
  };

  return (
    <div className = "mx-auto w-1/2 items-center m-6">
        <h2 className="text-3xl mb-3 font-bold tracking-tight">Fill this form to apply for a credit card</h2>
        <form onSubmit={handleSubmit} className="credit-form">
        <div className = "m-2">
          <label className = "m-2" htmlFor="CODE_GENDER">Gender:</label>
          <select
            id="CODE_GENDER"
            name="CODE_GENDER"
            value={formData.CODE_GENDER}
            onChange={handleChange}
            required
          >
            <option value="">Select Gender</option>
            <option value="M">Male</option>
            <option value="F">Female</option>
          </select>
          {formErrors.CODE_GENDER && <div className="error">{formErrors.CODE_GENDER}</div>}
        </div>

        <div className = "m-2">
          <label className = "m-2" htmlFor="FLAG_OWN_CAR">Own Car:</label>
          <select
            id="FLAG_OWN_CAR"
            name="FLAG_OWN_CAR"
            value={formData.FLAG_OWN_CAR}
            onChange={handleChange}
            required
          >
            <option value="">Select Car Ownership</option>
            <option value="Y">Yes</option>
            <option value="N">No</option>
          </select>
          {formErrors.FLAG_OWN_CAR && <div className="error">{formErrors.FLAG_OWN_CAR}</div>}
        </div>

        <div className = "m-2">
          <label className = "m-2" htmlFor="FLAG_OWN_REALTY">Own Realty:</label>
          <select
            id="FLAG_OWN_REALTY"
            name="FLAG_OWN_REALTY"
            value={formData.FLAG_OWN_REALTY}
            onChange={handleChange}
            required
          >
            <option value="">Select Realty Ownership</option>
            <option value="Y">Yes</option>
            <option value="N">No</option>
          </select>
          {formErrors.FLAG_OWN_REALTY && <div className="error">{formErrors.FLAG_OWN_REALTY}</div>}
        </div>

        <div className = "m-2">
          <label className = "m-2" htmlFor="CNT_CHILDREN">Number of Children:</label>
          <input className = "border"
            type="number"
            id="CNT_CHILDREN"
            name="CNT_CHILDREN"
            value={formData.CNT_CHILDREN}
            onChange={handleChange}
            required
          />
          {formErrors.CNT_CHILDREN && <div className="error">{formErrors.CNT_CHILDREN}</div>}
        </div>

        <div className = "m-2">
          <label className = "m-2" htmlFor="AMT_INCOME_TOTAL">Amount of income</label>
          <input className = "border"
            type="number"
            id="AMT_INCOME_TOTAL"
            name="AMT_INCOME_TOTAL"
            value={formData.AMT_INCOME_TOTAL}
            onChange={handleChange}
            required
          />
        </div>

        <div className = "m-2">
          <label className = "m-2" htmlFor="NAME_INCOME_TYPE">Income type</label>
          <input className = "border"
            type="string"
            id="NAME_INCOME_TYPE"
            name="NAME_INCOME_TYPE"
            value={formData.NAME_INCOME_TYPE}
            onChange={handleChange}
            required
          />
        </div>

        <div className = "m-2">
          <label className = "m-2" htmlFor="NAME_EDUCATION_TYPE">Education Type</label>
          <input className = "border"
            type="string"
            id="NAME_EDUCATION_TYPE"
            name="NAME_EDUCATION_TYPE"
            value={formData.NAME_EDUCATION_TYPE}
            onChange={handleChange}
            required
          />
        </div>
        <div className = "m-2">
          <label className = "m-2" htmlFor="NAME_FAMILY_STATUS">Family Status</label>
          <input className = "border"
            type="string"
            id="NAME_FAMILY_STATUS"
            name="NAME_FAMILY_STATUS"
            value={formData.NAME_FAMILY_STATUS}
            onChange={handleChange}
            required
          />
        </div>
        <div className = "m-2">
          <label className = "m-2" htmlFor="NAME_HOUSING_TYPE">Housing Type</label>
          <input className = "border"
            type="string"
            id="NAME_HOUSING_TYPE"
            name="NAME_HOUSING_TYPE"
            value={formData.NAME_HOUSING_TYPE}
            onChange={handleChange}
            required
          />
        </div>

        <div className = "m-2">
          <label className = "m-2" htmlFor="DAYS_BIRTH">Age in days</label>
          <input className = "border"
            type="number"
            id="DAYS_BIRTH"
            name="DAYS_BIRTH"
            value={formData.DAYS_BIRTH}
            onChange={handleChange}
            required
          />
        </div>
        <div className = "m-2">
          <label className = "m-2" htmlFor="DAYS_EMPLOYED">Days employed</label>
          <input className = "border"
            type="number"
            id="DAYS_EMPLOYED"
            name="DAYS_EMPLOYED"
            value={formData.DAYS_EMPLOYED}
            onChange={handleChange}
            required
          />
        </div>
        <div className = "m-2">
          <label className = "m-2" htmlFor="">Do you have a phone</label>
          <select
            id="FLAG_PHONE"
            name="FLAG_PHONE"
            value={formData.FLAG_PHONE}
            onChange={handleChange}
            required
          >
            <option value="0">Yes</option>
            <option value="1">No</option>
          </select>
        </div>
        <div className = "m-2">
          <label className = "m-2" htmlFor="">Do you have a work phone</label>
          <select
            id="FLAG_WORK_PHONE"
            name="FLAG_WORK_PHONE"
            value={formData.FLAG_WORK_PHONE}
            onChange={handleChange}
            required
          >
            <option value="0">Yes</option>
            <option value="1">No</option>
          </select>
        </div>
        <div className = "m-2">
          <label className = "m-2" htmlFor="">Do you have a mobile?</label>
          <select
            id="FLAG_MOBIL"
            name="FLAG_MOBIL"
            value={formData.FLAG_MOBIL}
            onChange={handleChange}
            required
          >
            <option value="0">Yes</option>
            <option value="1">No</option>
          </select>
        </div>
        {/* Add similar input fields for other properties in the CreditApprover class */}
        <div className = "m-2">
          <label className = "m-2" htmlFor="">Do you have an e-mail?</label>
          <select
            id="FLAG_EMAIL"
            name="FLAG_EMAIL"
            value={formData.FLAG_EMAIL}
            onChange={handleChange}
            required
          >
            <option value="0">Yes</option>
            <option value="1">No</option>
          </select>
        </div>
        <div className = "m-2">
          <label className = "m-2" htmlFor="">Occupation type</label>
          <input className = "border"
            id="OCCUPATION_TYPE"
            name="OCCUPATION_TYPE"
            type = "string"
            value={formData.OCCUPATION_TYPE}
            onChange={handleChange}
            required
          >
          </input>
        </div>
        <div className = "m-2">
          <label className = "m-2" htmlFor="">Number of Family members</label>
          <input className = "border"
            id="CNT_FAM_MEMBERS"
            name="CNT_FAM_MEMBERS"
            type = "number"
            value={formData.CNT_FAM_MEMBERS}
            onChange={handleChange}
            required
          >
          </input>
        </div>
        
        <Button type="submit">Submit</Button>
      </form>    </div>
    
  );
};


